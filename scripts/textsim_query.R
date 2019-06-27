#!/usr/bin/env Rscript

# Run query on TEXTSIM database on WMP. Save query results as CSV
# Example run
  # Rscript textsim_query.R query_fb_ad_100k.txt fb_ad100k.csv

library(readr)
library(RMySQL)

##--------------
##  Read inputs + query
##-----------------
args = commandArgs(trailingOnly=TRUE)
# Set defaults
if (length(args)==0) {
  args[1] = "my_query.txt" # default input file
  args[2] = "my_results.csv" # default output file
} else if (length(args)==1) {
  args[2] = "my_results.csv" # default output file
}

query_fp = args[1]
out_fp = args[2]

my_query <- read_lines(query_fp)
my_query <- paste0(my_query, collapse = " ")

msg <- paste("Query being run is in:", query_fp)
print(msg)
print(my_query)

##-----------------
## Query Textsim
##----------------
conn <- dbConnect(RMySQL::MySQL(),
                  host = "localhost",
                  dbname = "textsim",
                  user = "wmp_student",
                  password = "facebook==reports")

# submit query and retrieve all rows
rez <- dbGetQuery(conn, my_query)
write_csv(rez, path = out_fp)

out_msg <- paste("Query results saved as:", out_fp)
print(out_msg)

dbDisconnect(conn)




