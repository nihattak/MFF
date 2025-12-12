#' Compute Weights for K-means-Based Membership Assignment (Internal)
#'
#' This internal helper function converts hard K-means cluster assignments
#' into a simple membership weight matrix. Each element of the input vector
#' represents the cluster index assigned to a model. The function computes
#' weights based on cluster frequencies, assigning each model a weight
#' inversely proportional to the size of its cluster.
#'
#' @description
#' For a cluster assignment vector \code{clusters}, the weight for model \code{i}
#' is defined as:
#'
#' \deqn{ w_i = 1 / (\text{size of cluster assigned to } i) }
#'
#' This results in:
#' \itemize{
#'   \item Models in large clusters receive smaller weights.
#'   \item Models in small clusters receive larger weights.
#' }
#'
#' A membership matrix with dimensions \code{n × k} is produced, where:
#' \itemize{
#'   \item \code{n}: number of models
#'   \item \code{k}: number of clusters
#' }
#'
#' For row \code{i} and cluster \code{clusters[i]}, the weight matrix entry
#' \code{[i, clusters[i]]} equals \code{w_i}. All other entries in row \code{i}
#' are zero.
#'
#' @param clusters Integer vector containing the K-means cluster assignments for
#'        each model. Values must be in \code{1:k}, where \code{k} is the number
#'        of clusters.
#'
#' @return A numeric matrix of size \code{n × k} containing membership weights
#'         derived from cluster frequencies. Row names are preserved if the input
#'         vector has names.
#'
#' @details
#' This function is intended to provide soft membership-like weights for
#' K-means-based meta-modeling (MKF). Unlike fuzzy methods, weights do not sum
#' to 1 across each row; they directly reflect inverse cluster frequencies.
#'
#' @keywords internal
mkf_weight <- function(clusters) {
  # cluster_sizes her bir clusterin kaç tane olduğunu hesaplar
  # weights ağırlıklandırır ama seçimi vektörden fazla yapınca
  # c(1,2,2,2,3) 3 tane clusterlı bir model için o da ona göre atamaları yapar yani tekrarlı seçer
  # weight_matrixte ise [cbind(seq_len(n), clusters)] [i,j] şeklinde çalışır aşağıdaki for
  # for (i in 1:n) {
  # weight_matrix[i, clusters[i]] <- weights[i]
  # }
  # if'te de model isimleri matrise yeniden atanır eğer varsa ve çıktı verilir.

  k <- max(clusters)
  n <- length(clusters)

  cluster_sizes <- tabulate(clusters, nbins = k)

  weights <- 1 / cluster_sizes[clusters]

  weight_matrix <- matrix(0, nrow = n, ncol = k)
  weight_matrix[cbind(seq_len(n), clusters)] <- weights

  if (!is.null(names(clusters)))
    rownames(weight_matrix) <- names(clusters)

  weight_matrix
}
