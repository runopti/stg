#' Check whether STG Python package is available and can be loaded
#' 
#' This is used to avoid running tests on CRAN
#' 
#' @export
pystg_is_available <- function() {
  tryCatch({
    reticulate::import("stg")$STG
  },
  error = function(e) {
    FALSE
  }
  )
}

#' Install MAGIC Python Package
#'
#' Install MAGIC Python package into a virtualenv or conda env.
#'
#' On Linux and OS X the "virtualenv" method will be used by default
#' ("conda" will be used if virtualenv isn't available). On Windows,
#' the "conda" method is always used.
#'
#' @param envname Name of environment to install packages into
#' @param method Installation method. By default, "auto" automatically finds
#' a method that will work in the local environment. Change the default to
#' force a specific installation method. Note that the "virtualenv" method
#' is not available on Windows.
#' @param conda Path to conda executable (or "auto" to find conda using the PATH
#'  and other conventional install locations).
#' @param pip Install from pip, if possible.
#' @param ... Additional arguments passed to conda_install() or
#' virtualenv_install().
#'
#' @export
install.stg <- function(envname = "r-reticulate", method = "auto",
                          conda = "auto", pip=TRUE, ...) {
  message("Attempting to install STG python package with reticulate")
  tryCatch({
    reticulate::py_install("stg",
      envname = envname, method = method,
      conda = conda, pip=pip, ...
    )
    message("Install complete. Please restart R and try again.")
  },
  error = function(e) {
    stop(paste0(
      "Cannot locate STG Python package, please install through pip ",
      "(e.g. pip install stg) and then restart R."
    ))
  }
  )
}

pystg <- NULL