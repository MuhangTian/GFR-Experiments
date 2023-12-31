# NOTE: File to run AutoScore, mostly copied from original AutoScore github: https://github.com/nliulab/AutoScore/blob/master/R/AutoScore.R, 
# but with a few buggy lines removed in order to run, and changed some print statements in order to compare with our method (like calculation of AUPRC)
library(AutoScore)
library(glue)
library(pROC)
library(PRROC)
library(ggplot2)

#' @title AutoScore STEP(iii): Generate the initial score with the final list of variables (Re-run AutoScore Modules 2+3)
#' @param train_set A processed \code{data.frame} that contains data to be analyzed, for training.
#' @param validation_set A processed \code{data.frame} that contains data for validation purpose.
#' @param final_variables A vector containing the list of selected variables, selected from Step(ii)\code{\link{AutoScore_parsimony}}. Run \code{vignette("Guide_book", package = "AutoScore")} to see the guidebook or vignette.
#' @param max_score Maximum total score (Default: 100).
#' @param categorize  Methods for categorize continuous variables. Options include "quantile" or "kmeans" (Default: "quantile").
#' @param quantiles Predefined quantiles to convert continuous variables to categorical ones. (Default: c(0, 0.05, 0.2, 0.8, 0.95, 1)) Available if \code{categorize = "quantile"}.
#' @param max_cluster The max number of cluster (Default: 5). Available if \code{categorize = "kmeans"}.
#' @param metrics_ci whether to calculate confidence interval for the metrics of sensitivity, specificity, etc.
#' @return Generated \code{cut_vec} for downstream fine-tuning process STEP(iv) \code{\link{AutoScore_fine_tuning}}.
#' @references
#' \itemize{
#'  \item{Xie F, Chakraborty B, Ong MEH, Goldstein BA, Liu N. AutoScore: A Machine Learning-Based Automatic Clinical Score Generator and
#'   Its Application to Mortality Prediction Using Electronic Health Records. JMIR Medical Informatics 2020;8(10):e21798}
#' }
#' @seealso \code{\link{AutoScore_rank}}, \code{\link{AutoScore_parsimony}}, \code{\link{AutoScore_fine_tuning}}, \code{\link{AutoScore_testing}}, Run \code{vignette("Guide_book", package = "AutoScore")} to see the guidebook or vignette.
#' @export
#' @import pROC ggplot2
AutoScore_weighting_here <- function(train_set, validation_set, final_variables, max_score = 100, categorize = "quantile", max_cluster = 5, quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1), metrics_ci = FALSE) {
    # prepare train_set and Validation Set
    cat("****Included Variables: \n")
    print(data.frame(variable_name = final_variables))
    train_set_1 <- train_set[, c(final_variables, "label")]
    validation_set_1 <-
      validation_set[, c(final_variables, "label")]

    # AutoScore Module 2 : cut numeric and transfer categories and generate "cut_vec"
    cut_vec <- get_cut_vec(train_set_1, categorize = categorize, quantiles = quantiles, max_cluster = max_cluster)
    train_set_2 <- transform_df_fixed(train_set_1, cut_vec)
    validation_set_2 <- transform_df_fixed(validation_set_1, cut_vec)

    # AutoScore Module 3 : Score weighting
    score_table <- compute_score_table(train_set_2, max_score, final_variables)
    # cat("****Initial Scores: \n")
    # print(as.data.frame(score_table))
    # print_scoring_table(scoring_table = score_table, final_variable = final_variables)

    # Using "assign_score" to generate score based on new dataset and Scoring table "score_table"
    validation_set_3 <- assign_score(validation_set_2, score_table)
    validation_set_3$total_score <-
      rowSums(subset(validation_set_3, select = names(validation_set_3)[names(validation_set_3) !=
                                                                          "label"]))
    y_validation <- validation_set_3$label

    # Intermediate evaluation based on Validation Set
    # plot_roc_curve(validation_set_3$total_score, as.numeric(y_validation) - 1)
    # cat("***Performance (based on validation set):\n")
    # cat(
    #   "***The cutoffs of each variable generated by the AutoScore are saved in cut_vec. You can decide whether to revise or fine-tune them \n"
    # )
    model_complexity <- sum(sapply(cut_vec, function(x) length(x) + 1))
    # print(cut_vec)
    print(paste("Model Complexity: ", model_complexity))
    return(cut_vec)
  }

AutoScore_fine_tuning_here <- function(train_set, validation_set, final_variables, cut_vec, max_score = 100, metrics_ci = FALSE) {
    # Prepare train_set and Validation Set
    train_set_1 <- train_set[, c(final_variables, "label")]
    validation_set_1 <-
      validation_set[, c(final_variables, "label")]

    # AutoScore Module 2 : cut numeric and transfer categories (based on fix "cut_vec" vector)
    train_set_2 <-
      transform_df_fixed(train_set_1, cut_vec = cut_vec)
    validation_set_2 <-
      transform_df_fixed(validation_set_1, cut_vec = cut_vec)

    # AutoScore Module 3 : Score weighting
    score_table <-
      compute_score_table(train_set_2, max_score, final_variables)
    # cat("***Fine-tuned Scores: \n")
    # print(as.data.frame(score_table))
    # print_scoring_table(scoring_table = score_table, final_variable = final_variables)

    # Using "assign_score" to generate score based on new dataset and Scoring table "score_table"
    validation_set_3 <- assign_score(validation_set_2, score_table)
    validation_set_3$total_score <-
      rowSums(subset(validation_set_3, select = names(validation_set_3)[names(validation_set_3) !=
                                                                          "label"])) ## which name ="label"
    y_validation <- validation_set_3$label

    # Intermediate evaluation based on Validation Set after fine-tuning
    # plot_roc_curve(validation_set_3$total_score, as.numeric(y_validation) - 1)
    # cat("***Performance (based on validation set, after fine-tuning):\n")
    # print_roc_performance(y_validation, validation_set_3$total_score, threshold = "best", metrics_ci = metrics_ci)
    return(score_table)
  }

AutoScore_testing_here <- function(test_set, final_variables, cut_vec, scoring_table, threshold = "best", with_label = TRUE, metrics_ci = TRUE) {
    if (with_label) {
      # prepare test set: categorization and "assign_score"
      test_set_1 <- test_set[, c(final_variables, "label")]
      test_set_2 <-
        transform_df_fixed(test_set_1, cut_vec = cut_vec)
      test_set_3 <- assign_score(test_set_2, scoring_table)
      test_set_3$total_score <-
        rowSums(subset(test_set_3, select = names(test_set_3)[names(test_set_3) !=
                                                                "label"]))
      test_set_3$total_score[which(is.na(test_set_3$total_score))] <-
        0
      y_test <- test_set_3$label

      # Final evaluation based on testing set
      # plot_roc_curve(test_set_3$total_score, as.numeric(y_test) - 1)
      cat("********* TESTING PERFORMANCE *********\n")
      model_roc <- roc(y_test, test_set_3$total_score, quiet = T)
      print_roc_performance(y_test, test_set_3$total_score, threshold = threshold, metrics_ci = metrics_ci)
      #Modelprc <- pr.curve(test_set_3$total_score[which(y_test == 1)],test_set_3$total_score[which(y_test == 0)],curve = TRUE)
      #values<-coords(model_roc, "best", ret = c("specificity", "sensitivity", "accuracy", "npv", "ppv", "precision"), transpose = TRUE)
      pred_score <-
        data.frame(pred_score = test_set_3$total_score, Label = y_test)
      return(pred_score)

    } else {
      test_set_1 <- test_set[, c(final_variables)]
      test_set_2 <-
        transform_df_fixed(test_set_1, cut_vec = cut_vec)
      test_set_3 <- assign_score(test_set_2, scoring_table)
      test_set_3$total_score <-
        rowSums(subset(test_set_3, select = names(test_set_3)[names(test_set_3) !=
                                                                "label"]))
      test_set_3$total_score[which(is.na(test_set_3$total_score))] <-
        0
      pred_score <-
        data.frame(pred_score = test_set_3$total_score, Label = NA)
      return(pred_score)
    }
  }

print_roc_performance <- function(label, score, threshold = "best", metrics_ci = FALSE) {
    if (sum(is.na(score)) > 0)
      warning("NA in the score: ", sum(is.na(score)))
    model_roc <- roc.curve(scores.class0=score, weights.class0=label)
    model_prc <- pr.curve(scores.class0=score, weights.class0=label)
    auroc <- as.numeric(model_roc$auc)
    # if (as.numeric(auroc) >= 0.5){
    #   print("Treat normally")
    #   model_prc <- pr.curve(scores.class0 = score[label == 1], scores.class1 = score[label == 0])
    # }else{    # if less than, switch the prediction
    #   print("Switch prediction since worse than random guessing!")
    #   model_prc <- pr.curve(scores.class0 = score[label == 0], scores.class1 = score[label == 1])
    #   model_roc <- roc.curve(scores.class0 = score[label == 0], scores.class1 = score[label == 1])  
    #   auroc <- as.numeric(model_roc$auc)
    # }
    auprc <- as.numeric(model_prc$auc.integral)
    print(paste("AUROC:", round(auroc, 3)))
    print(paste("AUPRC:", round(auprc, 3)))
  }

#' @title Internal function: Calculate cut_vec from the training set (AutoScore Module 2)
#' @param df training set to be used for calculate the cut vector
#' @param categorize  Methods for categorize continuous variables. Options include "quantile" or "kmeans" (Default: "quantile").
#' @param quantiles Predefined quantiles to convert continuous variables to categorical ones. (Default: c(0, 0.05, 0.2, 0.8, 0.95, 1)) Available if \code{categorize = "quantile"}.
#' @param max_cluster The max number of cluster (Default: 5). Available if \code{categorize = "kmeans"}.
#' @return cut_vec for \code{transform_df_fixed}
get_cut_vec <- function(df, quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1), max_cluster = 5, categorize = "quantile") {
    # Generate cut_vec for downstream usage
    cut_vec <- list()

    for (i in setdiff(names(df), c("label", "label_time", "label_status"))) {
      # for factor variable
      if (is.factor(df[, i])) {
        if (length(levels(df[, i])) < 10)
          #(next)() else stop("ERROR: The number of categories should be less than 10")
          (next)()
        else
          warning("WARNING: The number of categories should be less than 10",
                  i)
      }

      ## mode 1 - quantiles
      if (categorize == "quantile") {
        # options(scipen = 20)
        #print("in quantile")
        cut_off_tmp <- quantile(df[, i], quantiles, na.rm=T)
        cut_off_tmp <- unique(cut_off_tmp)
        cut_off <- signif(cut_off_tmp, 3)  # remain 3 digits
        #print(cut_off)

        ## mode 2 k-means clustering
      } else if (categorize == "k_means") {
        #print("using k-means")
        clusters <- kmeans(na.omit(df[, i]), max_cluster)
        cut_off_tmp <- c()
        for (j in unique(clusters$cluster)) {
          #print(min(df[,i][clusters$cluster==j]))
          #print(length(df[,i][clusters$cluster==j]))
          cut_off_tmp <-
            append(cut_off_tmp, min(df[, i][clusters$cluster == j], na.rm=T))
          #print(cut_off_tmp)
        }
        cut_off_tmp <- append(cut_off_tmp, max(df[, i], na.rm=T))
        cut_off_tmp <- sort(cut_off_tmp)
        #print(names(df)[i])
        #assert (length(cut_off_tmp) == 6)
        cut_off_tmp <- unique(cut_off_tmp)
        cut_off <- signif(cut_off_tmp, 3)
        cut_off <- unique(cut_off)
        #print (cut_off)

      } else {
        stop('ERROR: please specify correct method for categorizing:  "quantile" or "k_means".')
      }

      l <- list(cut_off)
      names(l)[1] <- i
      cut_vec <- append(cut_vec, l)
      #print("****************************cut_vec*************************")
      #print(cut_vec)
    }
    ## delete min and max for each cut-off (min and max will be captured in the new dataset)
    if (length(cut_vec) != 0) { ## in case all the variables are categorical
      for (i in 1:length(cut_vec)) {
        if (length(cut_vec[[i]]) <= 2)
          cut_vec[[i]] <- c("let_binary")
        else
          cut_vec[[i]] <- cut_vec[[i]][2:(length(cut_vec[[i]]) - 1)]
      }
    }
    return(cut_vec)

  }
  
compute_score_table <- function(train_set_2, max_score, variable_list) {
    #AutoScore Module 3 : Score weighting
    # First-step logistic regression
    model <-
      glm(label ~ ., family = binomial(link = "logit"), data = train_set_2)
    coef_vec <- coef(model)
    if (length(which(is.na(coef_vec))) > 0) {
      warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
      coef_vec[which(is.na(coef_vec))] <- 1
    }
    train_set_2 <- change_reference(train_set_2, coef_vec)

    # Second-step logistic regression
    model <-
      glm(label ~ ., family = binomial(link = "logit"), data = train_set_2)
    coef_vec <- coef(model)
    if (length(which(is.na(coef_vec))) > 0) {
      warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
      coef_vec[which(is.na(coef_vec))] <- 1
    }

    # rounding for final scoring table "score_table"
    coef_vec_tmp <- round(coef_vec / min(coef_vec[-1]))
    score_table <- add_baseline(train_set_2, coef_vec_tmp)

    # normalization according to "max_score" and regenerate score_table
    total_max <- max_score
    total <- 0
    for (i in 1:length(variable_list))
      total <-
      total + max(score_table[grepl(variable_list[i], names(score_table))])
    score_table <- round(score_table / (total / total_max))
    return(score_table)
  }

change_reference <- function(df, coef_vec) {
  # delete label first
  df_tmp <- subset(df, select = names(df)[!names(df) %in% c("label", "label_time", "label_status")])
  names(coef_vec) <- gsub("[`]", "", names(coef_vec)) # remove the possible "`" in the names

  # one loops to go through all variable
  for (i in (1:length(df_tmp))) {
    var_name <- names(df_tmp)[i]
    var_levels <- levels(df_tmp[, i])
    var_coef_names <- paste0(var_name, var_levels)
    coef_i <- coef_vec[which(names(coef_vec) %in% var_coef_names)]
    # if min(coef_tmp)<0, the current lowest one will be used for reference
    if (min(coef_i) < 0) {
      ref <-
        var_levels[which(var_coef_names == names(coef_i)[which.min(coef_i)])]
      df_tmp[, i] <- relevel(df_tmp[, i], ref = ref)
    }
    # char_tmp <- paste("^", names(df_tmp)[i], sep = "")
    # coef_tmp <- coef_vec[grepl(char_tmp, names(coef_vec))]
    # coef_tmp <- coef_tmp[!is.na(coef_tmp)]

    # if min(coef_tmp)<0, the current lowest one will be used for reference
    # if (min(coef_tmp) < 0) {
    #   ref <- gsub(names(df_tmp)[i], "", names(coef_tmp)[which.min(coef_tmp)])
    #   df_tmp[, i] <- relevel(df_tmp[, i], ref = ref)
    # }
  }

  # add label again
  if(!is.null(df$label))  df_tmp$label <- df$label
  else{
  df_tmp$label_time <- df$label_time
  df_tmp$label_status <- df$label_status}
  return(df_tmp)
}

add_baseline <- function(df, coef_vec) {
  names(coef_vec) <- gsub("[`]", "", names(coef_vec)) # remove the possible "`" in the names
  df <- subset(df, select = names(df)[!names(df) %in% c("label", "label_time", "label_status")])
  coef_names_all <- unlist(lapply(names(df), function(var_name) {
    paste0(var_name, levels(df[, var_name]))
  }))
  coef_vec_all <- numeric(length(coef_names_all))
  names(coef_vec_all) <- coef_names_all
  # Remove items in coef_vec that are not meant to be in coef_vec_all
  # (i.e., the intercept)
  coef_vec_core <-
    coef_vec[which(names(coef_vec) %in% names(coef_vec_all))]
  i_coef <-
    match(x = names(coef_vec_core),
          table = names(coef_vec_all))
  coef_vec_all[i_coef] <- coef_vec_core
  coef_vec_all
}

assign_score <- function(df, score_table) {
  for (i in setdiff(names(df), c("label", "label_time", "label_status"))) {
    score_table_tmp <-
      score_table[grepl(i, names(score_table))]
    df[, i] <- as.character(df[, i])
    for (j in 1:length(names(score_table_tmp))) {
      df[, i][df[, i] %in% gsub(i, "", names(score_table_tmp)[j])] <-
        score_table_tmp[j]
    }

    df[, i] <- as.numeric(df[, i])
  }

  return(df)
}


args <- commandArgs(trailingOnly = TRUE)

if (args[1] == "eicu"){
  print("----- eICU -----")
  train_path <- "src/exp_6.6_to_6.27/data/MIMIC-WHOLE.csv"
  test_path <- "src/exp_6.6_to_6.27/data/eICU-union.csv"
}else if (as.numeric(args[1]) >= 1 & as.numeric(args[1]) <= 5){    # do MIMIC
  print("----- MIMIC -----")
  print(paste("FOLD", args[1]))
  train_path <- paste0("src/exp_6.6_to_6.27/data/k-fold/TRAIN-union-features-fold", args[1], ".csv")
  test_path <- paste0("src/exp_6.6_to_6.27/data/k-fold/TEST-union-features-fold", args[1], ".csv")
}else{
  stop("Args must be eicu (out of distribution testing), or 1-5 (folds on MIMIC)")
}

print("****** PATHS ******")
print(paste("Training:", train_path))
print(paste("Testing:", test_path))

train = read.csv(train_path)
test = read.csv(test_path)
names(train)[names(train) == "hospital_expire_flag"] <- "label"
names(test)[names(test) == "hospital_expire_flag"] <- "label"
train$label[train$label == -1] <- 0
test$label[test$label == -1] <- 0

set.seed(474)
out_split <- split_data(data = train, ratio = c(0.75, 0.25, 0), strat_by_label = FALSE)
train <- out_split$train_set
valid <- out_split$validation_set
print(paste("Training dataset size: ", dim(train)))
print(paste("Validation dataset size: ", dim(valid)))

ranking <- AutoScore_rank(train_set = train, method = "auc", validation_set = valid)

for (num_var in c(5, 10, 15, 14, 19, 20, 25, 30, 35, 40, 45)) {
  print(paste("================ Number of features:", num_var, "================="))
  start.time <- Sys.time()
  final_variables <- names(ranking[1:num_var])
  cut_vec <- AutoScore_weighting_here( 
    train_set = train, validation_set = valid,
    final_variables = final_variables, max_score = 100,
    categorize = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1)
  )
  scoring_table <- AutoScore_fine_tuning_here(
    train_set = train, validation_set = valid, 
    final_variables = final_variables, cut_vec = cut_vec, max_score = 100
  )
  pred_score <- AutoScore_testing_here(
    test_set = test, final_variables = final_variables, cut_vec = cut_vec,
    scoring_table = scoring_table, threshold = "best", with_label = TRUE
  )
  end.time <- Sys.time()
  time.taken <- difftime(end.time, start.time, units = "secs")
  print(paste("Time taken (seconds): ", time.taken))
  cat("**************************************************************\n")
}

