#' Check whether STG Python package is available and can be loaded
#' 
#' This is used to avoid running tests on CRAN
#' @return No return value, called for side effects
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

load_pystg <- function() {
  # load
  if (is.null(pystg)) {
    # first time load
    result <- try(pystg <<- reticulate::import("stg"))
  } else {
    # already loaded
    result <- try(reticulate::import("stg"))
  }
}


pystg <- NULL

