#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(tools)
library(ggh4x)
library(rjson)
library(fields)
library(ggplot2)
library(tikzDevice)
library(optparse)
library(gridExtra)
library(reshape2)
library(plyr)

g_legend <- function(ggobject) {
  # source: https://stackoverflow.com/a/13650878
  # extract legend from custom ggplot object
  tmp <- ggplot_gtable(ggplot_build(ggobject))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

post_process <- function(tex_file) {
  # plots post-processing
  no_ext_name <- gsub("\\.tex", "", tex_file)
  pdf_file <- paste0(no_ext_name, ".pdf")
  texi2pdf(tex_file,
    clean = TRUE,
    texi2dvi = Sys.which("lualatex")
  )
  file.remove(tex_file)
  file.rename(pdf_file, paste0("./docs/visuals/pdfs/", pdf_file))
  unlink(paste0(no_ext_name, "*.png"))
  unlink("Rplots.pdf")
}

visualize_train_spp_grid <- function(input_glob) {
  # find all csvs and declare variables to monitor
  events <- Sys.glob(file.path(input_glob, "events.csv"))
  training_log_watch <- c("accuracy.valid_accuracy", "loss.valid_loss")
  ensure_varying_args <- c("patterns", "tau_threshold", "seed")

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
      strip.background = element_blank(),
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      panel.grid = element_line(size = 1),
      axis.ticks.length = unit(.15, "cm"),
      axis.title.y = element_text(
        margin =
          margin(t = 0, r = 15, b = 0, l = 10)
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
    "train_spp_grid_patterns_tau_seed_",
    as.integer(as.POSIXct(Sys.time())), ".tex"
  )
  tikz(tex_file,
    width = 12, height = 6, standAlone = TRUE,
    engine = "luatex"
  )
  print(g)
  dev.off()
  post_process(tex_file)
}

# create option parser
parser <- OptionParser()
parser <- add_option(parser, c("-t", "--train-grid"),
  action = "store_true",
  default = FALSE,
  help = paste0(
    "Flag for plotting grid performance ",
    "with patterns, tau and seed being varied ",
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
  visualize_train_spp_grid(args$g)
}
