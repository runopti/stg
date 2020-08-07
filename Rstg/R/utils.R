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

failed_pystg_import <- function(e) {
  message("Error loading Python module stg")
  message(e)
  result <- as.character(e)
  if (length(grep("ModuleNotFoundError: No module named 'stg'", result)) > 0 ||
      length(grep("ImportError: No module named stg", result)) > 0) {
    # not installed
    if (utils::menu(c("Yes", "No"), title="Install STG Python package with reticulate?") == 1) {
      install.stg()
    }
  } else if (length(grep("r\\-reticulate", reticulate::py_config()$python)) > 0) {
    # installed, but envs sometimes give weird results
    message("Consider removing the 'r-reticulate' environment by running:")
    if (length(grep("virtualenvs", reticulate::py_config()$python)) > 0) {
      message("reticulate::virtualenv_remove('r-reticulate')")
    } else {
      message("reticulate::conda_remove('r-reticulate')")
    }
  }
}

load_pystg <- function() {
  delay_load = list(on_error=failed_pystg_import)
  # load
  if (is.null(pystg)) {
    # first time load
    result <- try(pystg <<- reticulate::import("stg", delay_load = delay_load))
  } else {
    # already loaded
    result <- try(reticulate::import("stg", delay_load = delay_load))
  }
}

#' Install STG Python Package
#'
#' Install STG Python package into a virtualenv or conda env.
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
      "Cannot locate STG Python package. Please install through pip ",
      "(e.g. pip install stg) and then restart R."
    ))
  }
  )
}


pystg <- NULL