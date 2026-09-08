# ============================================================================
# M731 Marketing Research — Course dataset generation
# ============================================================================
# Generates every simulated dataset used by the Session 1-13 slide decks and
# solutions walkthroughs, anchored to the Maritime Roast Coffee Co. running
# case (see Sources/.claude/agent-memory/m731-course-builder/running_case.md
# and course_datasets.md for the design rationale).
#
# Run this script from Sources/slides/data-raw/ (or source it with an
# absolute path). It writes every file to data-raw/output/; the reorg step
# then copies each file into the Session N/ folder(s) that use it.
#
# Reproducibility: every dataset uses a fixed seed derived from 731 so a
# fresh run reproduces bit-identical CSVs.
#
# Packages: base R + MASS (for mvrnorm, used in the survey factor structure)
# ============================================================================

if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("Package 'MASS' is required. Install with install.packages('MASS').")
}
library(MASS)

out_dir <- "output"  # relative to data-raw/ when this script is run from that folder
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat("Writing datasets to:", normalizePath(out_dir), "\n")

# ----------------------------------------------------------------------------
# 1. maritime_transactions.csv  (Session 1 — R basics / data frames)
# ----------------------------------------------------------------------------
set.seed(7311)

n_tx <- 15000
tiers <- c("Standard", "Signature", "Discovery")
tier_probs <- c(0.55, 0.30, 0.15)
tier_base_price <- c(Standard = 14.99, Signature = 22.50, Discovery = 32.00)

tx <- data.frame(
  order_id = seq_len(n_tx),
  subscriber_id = sample(1:4000, n_tx, replace = TRUE),
  order_date = as.Date("2026-01-01") + sample(0:181, n_tx, replace = TRUE),
  tier = sample(tiers, n_tx, replace = TRUE, prob = tier_probs),
  region = sample(c("Atlantic", "Ontario"), n_tx, replace = TRUE, prob = c(0.8, 0.2)),
  referral_source = sample(
    c("Organic", "Referral", "Social", "Search", "Retail_partner"),
    n_tx, replace = TRUE, prob = c(0.35, 0.15, 0.20, 0.20, 0.10)
  )
)

base_amt <- tier_base_price[tx$tier]
region_adj <- ifelse(tx$region == "Ontario", 1.05, 1.00)  # slightly higher AOV, smaller base
tx$amount <- round(pmax(5, base_amt * region_adj + rnorm(n_tx, 0, 2.5)), 2)

# 6-month churn flag: modestly more likely for Standard tier and Organic-only acquisition
churn_logit <- with(tx, -1.6 + 0.5 * (tier == "Standard") - 0.3 * (tier == "Discovery") +
                       0.25 * (referral_source == "Organic"))
tx$churn_flag <- rbinom(n_tx, 1, plogis(churn_logit))

write.csv(tx, file.path(out_dir, "maritime_transactions.csv"), row.names = FALSE)
cat("  maritime_transactions.csv:", nrow(tx), "rows\n")

# ----------------------------------------------------------------------------
# 2. statscan_coffee.csv  (Session 2 — secondary data / ggplot2)
# ----------------------------------------------------------------------------
set.seed(7312)

provinces <- c("Nova Scotia", "New Brunswick", "PEI", "Newfoundland",
               "Ontario", "Quebec", "Manitoba", "Saskatchewan",
               "Alberta", "British Columbia")
years <- 2019:2024

# Base 2019 household coffee spend by province + modest annual growth + noise
base_spend <- c(
  "Nova Scotia" = 310, "New Brunswick" = 295, "PEI" = 300, "Newfoundland" = 285,
  "Ontario" = 340, "Quebec" = 325, "Manitoba" = 300, "Saskatchewan" = 290,
  "Alberta" = 330, "British Columbia" = 350
)
growth_rate <- 0.035  # ~3.5% per year, illustrative

coffee <- expand.grid(province = provinces, year = years, stringsAsFactors = FALSE)
coffee <- coffee[order(coffee$province, coffee$year), ]
coffee$spending <- with(coffee, round(
  base_spend[province] * (1 + growth_rate) ^ (year - 2019) + rnorm(nrow(coffee), 0, 6),
  2
))

write.csv(coffee, file.path(out_dir, "statscan_coffee.csv"), row.names = FALSE)
cat("  statscan_coffee.csv:", nrow(coffee), "rows\n")
cat("  NOTE: illustrative figures modelled loosely on published StatsCan household\n")
cat("  spending patterns, not a direct extract of a specific StatsCan table.\n")

# ----------------------------------------------------------------------------
# 3 & core. maritime_survey.csv  (Sessions 3, 6, 7, 8, 9, 11, 12)
# ----------------------------------------------------------------------------
set.seed(7313)

n_sv <- 800

region <- sample(c("Atlantic", "Ontario_prospect"), n_sv, replace = TRUE, prob = c(0.65, 0.35))
subscriber_status <- sample(c("Current", "Lapsed"), n_sv, replace = TRUE, prob = c(0.82, 0.18))
tier <- sample(c("Standard", "Signature", "Discovery"), n_sv, replace = TRUE, prob = c(0.5, 0.32, 0.18))
age <- round(pmin(75, pmax(21, rnorm(n_sv, 39, 11))))
gender <- sample(c("F", "M", "Nonbinary"), n_sv, replace = TRUE, prob = c(0.52, 0.44, 0.04))
tenure_months <- round(pmax(1, rexp(n_sv, 1 / 18)))
tenure_months[subscriber_status == "Lapsed"] <- round(pmax(1, tenure_months[subscriber_status == "Lapsed"] * runif(sum(subscriber_status == "Lapsed"), 0.3, 0.8)))

# --- Brand-attachment battery (5 items, Likert 1-5), single factor, alpha ~0.85 ---
attach_factor <- rnorm(n_sv, 0, 1)
attach_loadings <- c(0.80, 0.78, 0.75, 0.72, 0.68)
brand_attach <- sapply(attach_loadings, function(l) {
  raw <- l * attach_factor + sqrt(1 - l^2) * rnorm(n_sv)
  scaled <- round(pmin(5, pmax(1, 3 + raw * 1.1)))
  scaled
})
colnames(brand_attach) <- paste0("brand_attach_", 1:5)
# Lapsed subscribers and longer-tenure "current" subscribers attach a bit more strongly on avg
attach_shift <- ifelse(subscriber_status == "Lapsed", -0.5, 0) + pmin(0.6, tenure_months / 60)
for (j in 1:5) {
  brand_attach[, j] <- pmin(5, pmax(1, round(brand_attach[, j] + attach_shift * sample(c(0,1), n_sv, TRUE, c(0.3,0.7)))))
}

# --- Needs battery (12 items, Likert 1-5), 3 correlated factors (4 items each) ---
# Factor scores: Convenience, Quality, Social — mildly correlated (r ~ .2-.35)
Sigma_needs <- matrix(c(1, .30, .20,
                         .30, 1, .25,
                         .20, .25, 1), nrow = 3)
needs_factors <- mvrnorm(n_sv, mu = c(0, 0, 0), Sigma = Sigma_needs)
colnames(needs_factors) <- c("conv_f", "qual_f", "soc_f")

needs_loadings <- c(0.75, 0.72, 0.70, 0.68)  # per-factor item loadings
needs <- matrix(NA_real_, n_sv, 12)
factor_map <- rep(1:3, each = 4)
for (j in 1:12) {
  f <- needs_factors[, factor_map[j]]
  l <- needs_loadings[((j - 1) %% 4) + 1]
  raw <- l * f + sqrt(1 - l^2) * rnorm(n_sv)
  needs[, j] <- pmin(5, pmax(1, round(3 + raw * 1.1)))
}
colnames(needs) <- paste0("needs_", 1:12)

# --- WTP (willingness to pay, $ per bag), influenced by region, tier, status ---
wtp <- with(list(),
  round(pmax(8, 21 +
    ifelse(region == "Ontario_prospect", -1.8, 0) +
    ifelse(tier == "Discovery", 4.5, ifelse(tier == "Signature", 2, 0)) +
    ifelse(subscriber_status == "Lapsed", -2.2, 0) +
    0.15 * needs_factors[, "qual_f"] * 3 +
    rnorm(n_sv, 0, 3.2)), 2)
)

# --- Satisfaction (1-10), driven by tenure, tier, needs factors, brand attachment ---
attach_mean <- rowMeans(brand_attach)
satisfaction_raw <- 6.0 +
  0.35 * scale(attach_mean)[, 1] +
  0.25 * scale(needs_factors[, "qual_f"])[, 1] +
  0.15 * scale(needs_factors[, "conv_f"])[, 1] +
  ifelse(tier == "Discovery", 0.4, ifelse(tier == "Standard", -0.3, 0)) +
  0.01 * pmin(tenure_months, 36) +
  ifelse(subscriber_status == "Lapsed", -1.1, 0) +
  rnorm(n_sv, 0, 0.9)
satisfaction <- round(pmin(10, pmax(1, satisfaction_raw)), 1)

# --- Preference for dark roast (associated loosely with region) ---
prefers_dark_roast <- rbinom(n_sv, 1, plogis(-0.2 + ifelse(region == "Atlantic", 0.35, -0.1)))
prefers_dark_roast <- factor(prefers_dark_roast, labels = c("No", "Yes"))

# --- Pre/post score demo pair (illustrative paired-t-test only; ~200 respondents) ---
has_prepost <- sample(c(TRUE, FALSE), n_sv, replace = TRUE, prob = c(0.25, 0.75))
pre_score <- rep(NA_real_, n_sv)
post_score <- rep(NA_real_, n_sv)
pre_score[has_prepost] <- round(rnorm(sum(has_prepost), 6.0, 1.2), 1)
post_score[has_prepost] <- round(pre_score[has_prepost] + rnorm(sum(has_prepost), 0.4, 0.9), 1)

survey <- data.frame(
  respondent_id = seq_len(n_sv),
  region = region,
  subscriber_status = subscriber_status,
  tier = tier,
  age = age,
  gender = gender,
  tenure_months = tenure_months,
  wtp = wtp,
  satisfaction = satisfaction,
  prefers_dark_roast = prefers_dark_roast,
  brand_attach,
  needs,
  pre_score = pre_score,
  post_score = post_score
)

write.csv(survey, file.path(out_dir, "maritime_survey.csv"), row.names = FALSE)
cat("  maritime_survey.csv:", nrow(survey), "rows,", ncol(survey), "cols\n")

# ----------------------------------------------------------------------------
# 4. kaggle_survey_raw.csv  (Session 4 — cleaning messy survey exports)
# ----------------------------------------------------------------------------
# NOTE: the original slide deck referenced Kaggle's "Young People Survey"
# (Sabo, 2016). That dataset is third-party and not ours to redistribute, so
# this is an instructor-simulated STAND-IN with the same messy structure
# (inconsistent column names, mixed NA encodings, a reversed pair, and two
# multi-item batteries) referenced by the Session 4 lab code. Document this
# substitution to students if using the real dataset is ever preferred.
set.seed(7314)

n_kg <- 1010

rand_na <- function(x, p = 0.05) {
  idx <- sample(seq_along(x), size = round(length(x) * p))
  x[idx] <- NA
  x
}

kg <- data.frame(
  `Age ` = rand_na(round(pmin(30, pmax(15, rnorm(n_kg, 20, 2.5)))), 0.03),
  `Gender` = sample(c("female", "male", "", "NA"), n_kg, replace = TRUE, prob = c(0.48, 0.46, 0.03, 0.03)),
  `Village - town` = sample(c("village", "city", NA), n_kg, replace = TRUE, prob = c(0.35, 0.6, 0.05)),
  check.names = FALSE
)

for (i in 1:8) {
  kg[[paste0("Music ", i)]] <- rand_na(sample(1:5, n_kg, replace = TRUE), 0.06)
}
for (i in 1:6) {
  kg[[paste0("Movies_", i)]] <- rand_na(sample(1:5, n_kg, replace = TRUE), 0.06)
}

# loneliness_1 and loneliness_2 are reverse-worded relative to each other
lone_factor <- rnorm(n_kg)
kg[["loneliness_1"]] <- rand_na(pmin(5, pmax(1, round(3 + 0.8 * lone_factor + rnorm(n_kg, 0, 0.8)))), 0.04)
kg[["loneliness_2"]] <- rand_na(pmin(5, pmax(1, round(3 - 0.8 * lone_factor + rnorm(n_kg, 0, 0.8)))), 0.04)  # reverse-worded

# A few straight-liners in the music battery to make the SD-based detection demo work
straightliners <- sample(seq_len(n_kg), 25)
for (i in 1:8) {
  kg[straightliners, paste0("Music ", i)] <- 3
}

write.csv(kg, file.path(out_dir, "kaggle_survey_raw.csv"), row.names = FALSE, na = "")
cat("  kaggle_survey_raw.csv:", nrow(kg), "rows,", ncol(kg), "cols (simulated stand-in)\n")

# ----------------------------------------------------------------------------
# 5. maritime_focus.txt  (Session 5 — qualitative / focus-group transcripts)
# ----------------------------------------------------------------------------
transcripts <- c(
"FOCUS GROUP 1 — Lapsed subscribers, Halifax, n=8
MODERATOR: Thanks everyone for coming in. Let's start broad -- tell us about your morning coffee habits.
P1: Coffee is non-negotiable for my morning routine. I need it before I talk to anyone.
P2: Same, it's part of the routine, but I ended up cancelling my Maritime Roast subscription because the price crept up.
P3: I loved the subscription at first, the discovery box was fun, but shipping got slow near the end.
MODERATOR: What made you finally cancel?
P4: Honestly the price increase felt sudden. I didn't get much warning.
P1: For me it was that I found a local roaster I could just walk to.
P2: The convenience mattered less once a cafe opened near my house.
MODERATOR: If you could tell the CEO one thing, what would it be?
P3: Be more transparent about price changes and give people a heads up.
P4: Bring back more flexible subscription pausing, not just full cancellation.",

"FOCUS GROUP 2 — Lapsed subscribers, Moncton, n=7
MODERATOR: Let's talk about your experience with subscription coffee generally.
P5: I subscribed to Maritime Roast for about a year. Loved the morning ritual of a fresh bag.
P6: I churned because I moved and the delivery timing became unpredictable.
P7: Price sensitivity was a factor for me too, especially after a rate increase.
MODERATOR: Tell me about a perfect morning with coffee.
P5: Quiet kitchen, good light, and a coffee I didn't have to think about because it just arrives.
P8: For me it's less about ritual and more about not running out -- convenience over everything.
MODERATOR: What would bring you back?
P6: Predictable delivery windows and maybe a loyalty discount for returning subscribers.
P7: A cheaper entry tier. The Discovery tier was too rich for my budget.",

"FOCUS GROUP 3 — Current subscribers considering downgrade, Halifax, n=8
MODERATOR: What keeps you subscribed to Maritime Roast?
P9: Quality of the beans, honestly. Consistently good, especially the Signature roast.
P10: The subscription discipline -- I never run out of coffee, which matters more than I expected.
P11: I like supporting a Halifax company, feels good for my morning routine.
MODERATOR: Any frustrations?
P12: Shipping cost went up and I'm thinking about downgrading from Discovery to Standard.
P9: I wish there was more flexibility on delivery frequency.
MODERATOR: What comes to mind when you hear 'Maritime Roast'?
P10: Reliability. Quality. A little bit of home.
P11: Local pride, honestly.",

"FOCUS GROUP 4 — Mixed current and lapsed, Ontario prospects, n=6
MODERATOR: Some of you are considering a coffee subscription for the first time. What matters most?
P13: Price transparency up front, no surprise increases.
P14: I want to know shipping times before I commit, especially since I'm in Ontario, further from Halifax.
P15: Quality has to compete with what I can get locally in Toronto.
MODERATOR: Would a subscription from an Atlantic Canada company appeal to you?
P16: If the quality story is compelling and shipping is reliable, sure.
P13: I like the idea of discovering something outside the usual big brands.
P14: Price sensitivity is real though -- if it's much more than local options I'd pass."
)

writeLines(paste(transcripts, collapse = "\n\n---\n\n"), file.path(out_dir, "maritime_focus.txt"))
cat("  maritime_focus.txt: 4 simulated focus-group transcripts\n")

# ----------------------------------------------------------------------------
# 6. maritime_abtest.csv  (Session 10 — A/B experiment; reused in Session 11)
# ----------------------------------------------------------------------------
set.seed(7316)

n_ab <- 2000
treatment <- sample(c("control", "new_offer"), n_ab, replace = TRUE)
age_ab <- round(pmin(75, pmax(21, rnorm(n_ab, 39, 11))))
gender_ab <- sample(c("F", "M", "Nonbinary"), n_ab, replace = TRUE, prob = c(0.52, 0.44, 0.04))
tenure_ab <- round(pmax(1, rexp(n_ab, 1 / 18)))

# Baseline upgrade rate ~10%; treatment lifts by ~6.5pp (within the 5-8pp target range)
base_logit <- -2.2 + 0.01 * tenure_ab
upgrade_prob <- plogis(base_logit + ifelse(treatment == "new_offer", 0.55, 0))
upgraded <- rbinom(n_ab, 1, upgrade_prob)

ab <- data.frame(
  subscriber_id = seq_len(n_ab),
  treatment = treatment,
  age = age_ab,
  gender = gender_ab,
  tenure_months = tenure_ab,
  upgraded = upgraded
)

write.csv(ab, file.path(out_dir, "maritime_abtest.csv"), row.names = FALSE)
cat("  maritime_abtest.csv:", nrow(ab), "rows; observed lift =",
    round(100 * (mean(ab$upgraded[ab$treatment == "new_offer"]) -
                 mean(ab$upgraded[ab$treatment == "control"])), 1), "pp\n")

cat("\nAll datasets written to", normalizePath(out_dir), "\n")
