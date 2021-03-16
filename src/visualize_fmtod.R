#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(ggplot2)
library(tikzDevice)
library(optparse)
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

visualize_fmtod <- function(train_labels, valid_labels, test_labels) {
  # read data, tabulate and add partition information
  train_labels <- as.data.frame(table(read.table(train_labels)))
  valid_labels <- as.data.frame(table(read.table(valid_labels)))
  test_labels <- as.data.frame(table(read.table(test_labels)))
  train_labels$Partition <- "Train"
  valid_labels$Partition <- "Validation"
  test_labels$Partition <- "Test"

  # clean data for plotting
  collections <- rbind(train_labels, valid_labels, test_labels)
  names(collections)[c(1, 2)] <- c("Class", "Frequency")
  collections[["Class"]] <- as.factor(collections[["Class"]])
  collections[["Partition"]] <- factor(collections[["Partition"]],
    levels = c("Train", "Validation", "Test")
  )

  # create ggplot object
  g <- ggplot(collections, aes(x = Class, y = Frequency, fill = Partition)) +
    geom_bar(
      stat = "identity", position = "dodge", alpha = 0.8,
      color = "black", size = 0.25
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = c(0.1, 0.85),
      legend.background = element_blank(),
      text = element_text(size = 22),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      panel.grid = element_line(size = 1),
      axis.title.y = element_text(
        margin =
          margin(t = 0, r = 10, b = 0, l = 0)
      ),
      axis.title.x = element_text(
        margin =
          margin(t = 10, r = 0, b = -5, l = 0)
      )
    ) +
    scale_fill_manual(values = c("cornflowerblue", "darkgreen", "orangered")) +
    ggtitle("Preprocessed FMTOD data distribution by class and partition")

  # plot object and convert to pdf via tikz
  tex_file <- paste0("fmtod_summary_statistics.tex")
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
parser <- add_option(parser,
  c(
    "-t",
    "--train-labels"
  ),
  type = "character",
  help = "Path to training data labels"
)
parser <- add_option(parser,
  c(
    "-v",
    "--valid-labels"
  ),
  type = "character",
  help = "Path to validation data labels"
)
parser <- add_option(parser,
  c(
    "-e",
    "--test-labels"
  ),
  type = "character",
  help = "Path to test/evaluation data labels"
)

# parse arguments and assign function
args <- parse_args(parser)
visualize_fmtod(
  args[["train-labels"]], args[["valid-labels"]],
  args[["test-labels"]]
)
