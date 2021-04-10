#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(ggh4x)
library(rjson)
library(ggplot2)
library(tikzDevice)
library(optparse)
library(reshape2)
library(plyr)
library(RColorBrewer)
library(tools)

post_process <- function(tex_file) {
  # plots post-processing
  no_ext_name <- gsub("\\.tex", "", tex_file)
  pdf_file <- paste0(no_ext_name, ".pdf")
  texi2pdf(tex_file,
    clean = TRUE,
    texi2dvi = Sys.which("lualatex")
  )
  file.remove(tex_file)
  file.rename(pdf_file, paste0("./docs/visuals/pdfs/generated/", pdf_file))
  unlink(paste0(no_ext_name, "*.png"))
  unlink("Rplots.pdf")
}

visualize_grid_train <- function(input_glob,
                                 training_log_watch =
                                   c(
                                     "accuracy.valid_accuracy",
                                     "loss.valid_loss"
                                   ),
                                 ensure_varying_args = c(
                                   "patterns",
                                   "tau_threshold", "seed"
                                 )) {
  # find all csvs and declare variables to monitor
  events <- Sys.glob(file.path(input_glob, "events.csv"))

  # accumulate all data into collections
  collections <- lapply(events, function(event) {
    model_log_directory <- dirname(dirname(event))
    model_config <- fromJSON(file = file.path(
      model_log_directory,
      "model_config.json"
    ))
    training_config <- fromJSON(file = file.path(
      model_log_directory,
      "training_config.json"
    ))
    event_log <- read.csv(event, stringsAsFactors = FALSE)[c(
      "steps",
      training_log_watch
    )]
    return(list(c(model_config, training_config), event_log))
  })

  # find varying arguments
  configs <- as.data.frame(do.call(rbind, lapply(collections, `[[`, 1)))
  varying_args <- unlist(sapply(names(configs), function(colname) {
    if (nrow(unique(configs[colname])) != 1) {
      return(colname)
    }
  }))
  if ("models_directory" %in% varying_args) {
    varying_args <- varying_args[-which(varying_args == "models_directory")]
  }

  # check to ensure sanity of arguments
  if (!setequal(varying_args, ensure_varying_args)) {
    stop(paste0(
      "Varying arguments are strictly different from patterns,",
      " tau_threshold and seed"
    ))
  }

  # create an aggregate object
  collections <- do.call(rbind, lapply(collections, function(collection) {
    relevant_args <- unlist(collection[[1]][varying_args])
    for (i in 1:length(relevant_args)) {
      collection[[2]][names(relevant_args)[i]] <- relevant_args[i]
    }
    return(collection[[2]])
  }))

  # compute convergence windows
  aggregate_mins <- aggregate(loss.valid_loss ~ patterns +
    tau_threshold + seed,
  data = collections, FUN = min
  )
  aggregate_mins <- match_df(collections, aggregate_mins)
  convergence_window <- do.call(data.frame, aggregate(steps ~ patterns +
    tau_threshold,
  data = aggregate_mins,
  FUN = function(x) {
    c(
      mean = mean(x),
      sd = sd(x)
    )
  }
  ))

  # change strings to factors in collections
  collections$patterns <- factor(collections$patterns)
  levels(collections$patterns) <- paste0(
    "$P=\\texttt{",
    gsub(
      "\\_", "\\\\_",
      levels(collections$patterns)
    ),
    "}$"
  )
  collections$tau_threshold <- factor(collections$tau_threshold)
  levels(collections$tau_threshold) <- paste0(
    "$\\mbox{\\Large$\\tau$}=",
    levels(collections$tau_threshold),
    "$"
  )
  collections$seed <- factor(collections$seed)

  # change strings to factors in convergence window
  convergence_window$patterns <- factor(convergence_window$patterns)
  levels(convergence_window$patterns) <- paste0(
    "$P=\\texttt{",
    gsub(
      "\\_", "\\\\_",
      levels(convergence_window$patterns)
    ),
    "}$"
  )
  convergence_window$tau_threshold <- factor(convergence_window$tau_threshold)
  levels(convergence_window$tau_threshold) <- paste0(
    "$\\mbox{\\Large$\\tau$}=",
    levels(convergence_window$tau_threshold),
    "$"
  )

  # add extra text to patterns
  levels(collections$patterns) <- paste0(
    c("Small\n", "Medium\n", "Large\n"),
    levels(collections$patterns)
  )
  levels(convergence_window$patterns) <- paste0(
    c("Small\n", "Medium\n", "Large\n"),
    levels(convergence_window$patterns)
  )

  # create ggplot object
  g <- ggplot() +
    geom_rect(
      data = convergence_window,
      aes(
        xmin = steps.mean - steps.sd, xmax = steps.mean + steps.sd,
        fill = "blue"
      ),
      ymin = -Inf, ymax = Inf, alpha = 0.1, color = "black", size = 0.1
    ) +
    geom_line(
      data = collections,
      aes(x = steps, y = accuracy.valid_accuracy, color = seed)
    ) +
    xlab("Updates") +
    ylab("Validation accuracy") +
    labs(color = "Random\nseed") +
    theme_bw() +
    theme(
      text = element_text(size = 22),
      strip.background = element_blank(),
      legend.position = "bottom",
      panel.grid = element_line(size = 1),
      axis.ticks.length = unit(.15, "cm"),
      axis.title.y = element_text(
        margin =
          margin(t = 0, r = 10, b = 0, l = 0)
      )
    ) +
    scale_color_brewer(palette = "Paired") +
    scale_fill_manual(
      name = "Convergence\nwindow",
      labels = c(""),
      values = c("blue")
    ) +
    facet_nested(tau_threshold ~ patterns)

  # plot object and convert to pdf via tikz
  tex_file <- paste0(
    "train_spp_grid_",
    as.integer(as.POSIXct(Sys.time())), ".tex"
  )
  tikz(tex_file,
    width = 12, height = 10, standAlone = TRUE,
    engine = "luatex"
  )
  print(g)
  dev.off()
  post_process(tex_file)
}

visualize_grid_evaluate <- function(input_glob,
                                    ensure_varying_args = c(
                                      "patterns",
                                      "tau_threshold",
                                      "seed"
                                    )) {
  # start collecting raw data
  model_directories <- Sys.glob(input_glob)
  collections <- lapply(model_directories, function(model_directory) {
    comparison <- fromJSON(file = Sys.glob(
      file.path(model_directory, "compare_*.json")
    ))
    distances <- colMeans(do.call(rbind, lapply(comparison[["comparisons"]], function(x) {
      return(c(
        x[["inter_model_distance_metrics"]][["softmax_difference_norm"]],
        x[["inter_model_distance_metrics"]][["binary_misalignment_rate"]]
      ))
    })))
    model_config <- fromJSON(file = Sys.glob(
      file.path(
        model_directory,
        "model_config.json"
      )
    ))
    seed <- fromJSON(file = Sys.glob(
      file.path(
        model_directory,
        "training_config.json"
      )
    ))[["seed"]]
    tau_threshold <- model_config[["tau_threshold"]]
    patterns <- model_config[["patterns"]]
    spp_acc <- fromJSON(file = Sys.glob(
      file.path(
        model_directory,
        "spp_*_classification_*.json"
      )
    ))[["accuracy"]]
    regex_acc <- fromJSON(file = Sys.glob(
      file.path(
        model_directory,
        "regex_*_classification_*.json"
      )
    ))[["accuracy"]]
    return(data.frame(
      patterns = patterns, tau_threshold = tau_threshold,
      seed = seed, spp_acc = spp_acc, regex_acc = regex_acc,
      softmax_distance = distances[1],
      binary_distance = distances[2]
    ))
  })

  # transform data frame to be nicer
  collections <- do.call(rbind, collections)
  collections[["tau_threshold"]] <- as.factor(collections[["tau_threshold"]])

  backup <- data.frame(collections)
  collections <- data.frame(backup)

  collections <- melt(collections, id.var = c("patterns", "tau_threshold", "seed"))

  # compute varying arguments here
  varying_args <- unlist(sapply(names(collections), function(colname) {
    if (nrow(unique(collections[colname])) != 1) {
      return(colname)
    }
  }))
  varying_args <- varying_args[which(varying_args != "variable" & varying_args != "value")]

  # check to ensure sanity of arguments
  if (!setequal(varying_args, ensure_varying_args)) {
    stop(paste0(
      "Varying arguments are strictly different from patterns,",
      " tau_threshold and seed"
    ))
  }

  # convert to mathematical notations
  collections$patterns <- factor(collections$patterns)
  levels(collections$patterns) <- paste0(
    "$P=\\texttt{",
    gsub(
      "\\_", "\\\\_",
      levels(collections$patterns)
    ),
    "}$"
  )
  collections$type <- factor(gsub(
    ".*\\_distance",
    "Distance metrics",
    gsub(
      ".*\\_acc", "Accuracy",
      collections$variable
    )
  ),
  levels = c("Accuracy", "Distance metrics")
  )
  collections$variable <- gsub(
    "spp\\_acc", "SoPa++",
    collections$variable
  )
  collections$variable <- gsub(
    "regex\\_acc", "RE\nproxy",
    collections$variable
  )
  collections$variable <- gsub(
    "softmax\\_distance", "$\\\\overline{\\\\delta_{\\\\sigma}}$",
    collections$variable
  )
  collections$variable <- gsub(
    "binary\\_distance", "$\\\\overline{\\\\delta_{b}}$",
    collections$variable
  )
  collections$variable <- factor(collections$variable, levels = c(
    "SoPa++",
    "RE\nproxy",
    "$\\overline{\\delta_{\\sigma}}$",
    "$\\overline{\\delta_{b}}$"
  ))

  # add extra text to patterns
  levels(collections$patterns) <- paste0(
    c("Small\n", "Medium\n", "Large\n"),
    levels(collections$patterns)
  )

  # make ggplot object
  g <- ggplot(collections, aes(x = tau_threshold, y = value, fill = variable)) +
    stat_boxplot(
      geom = "errorbar", width = 0.2,
      position = position_dodge(width = 1)
    ) +
    geom_boxplot(
      position = position_dodge(width = 1), outlier.shape = 1,
      outlier.size = 2
    ) +
    xlab("\\mbox{\\large$\\tau$}") +
    ylab("Metric") +
    labs(fill = "") +
    theme_bw() +
    theme(
      text = element_text(size = 22),
      strip.background = element_blank(),
      legend.position = "bottom",
      panel.grid = element_line(size = 1),
      axis.ticks.length = unit(.15, "cm"),
      axis.title.y = element_text(
        margin =
          margin(t = 0, r = 10, b = 0, l = 0)
      ),
      axis.title.x = element_text(
        margin =
          margin(t = 10, r = 0, b = -5, l = 0)
      )
    ) +
    scale_fill_manual(values = brewer.pal(6, "Paired")[c(5, 6, 1, 2)]) +
    facet_nested(type ~ patterns, scales = "free")

  # plot object and convert to pdf via tikz
  tex_file <- paste0(
    "evaluate_spp_grid_",
    as.integer(as.POSIXct(Sys.time())), ".tex"
  )
  tikz(tex_file,
    width = 12, height = 8, standAlone = TRUE,
    engine = "luatex"
  )
  print(g)
  dev.off()
  post_process(tex_file)
}

# create option parser
parser <- OptionParser()
parser <- add_option(parser,
  c(
    "-t",
    "--train-grid"
  ),
  action = "store_true",
  default = FALSE,
  help = paste0(
    "Flag for plotting grid training performance ",
    "with patterns, tau_threshold and seed being varied ",
    "[default: %default]"
  )
)
parser <- add_option(parser,
  c(
    "-e",
    "--evaluate-grid"
  ),
  action = "store_true",
  default = FALSE,
  help = paste0(
    "Flag for plotting evaluation relationships between ",
    "tau_threshold, patterns, performances and model pair distances ",
    "with patterns, tau_threshold and seed being varied ",
    "[default: %default]"
  )
)
parser <- add_option(parser, c("-g", "--glob"),
  type = "character",
  help = "Glob for finding input files"
)

# parse arguments and assign function
args <- parse_args(parser)
if (args$t) {
  visualize_grid_train(args$g)
} else if (args$e) {
  visualize_grid_evaluate(args$g)
}
