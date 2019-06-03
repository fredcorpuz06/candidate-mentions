library(tidyverse)
og_ads <- read_csv("./data/og_ad.csv")
cands <- read_csv("./data/candidate_images/cand_image_data.csv")




text_creatives <- og_ads %>% 
  select(ad_id, queries, funding_entity, ad_creative_body)
str_extract("Robert Kennedy, Jr.", "[^,.]*")

# Extract last names
cands2 <- cands %>% 
  select(cand_name, party) %>% 
  mutate(no_honorific = str_extract(cand_name, "[^,]*"),
         last_name = str_extract(no_honorific, "[A-Za-zÀ-ÿ']+?$"))



findMentions <- function(df_ad, identifiers, full_names) {
  # Finds mentions of a person's name in text.
  #
  # Returns:
  #   Vector of full names of mentioned politicians
  ad_text <- df_ad$ad_creative_body
  mentioned <- sapply(identifiers, function(x) str_detect(ad_text, x))
  mask <- which(unname(mentioned))
  hits <- identifiers[mask]
  mentions <- full_names[mask]
  n_rows <- length(mentions)
  df_ad$temp_id <- c(1)
  if(length(mask) == 0) {
    rez <- tibble(temp_id = rep(1, 1), 
                      hits = c(NA),
                      mentions = c(NA))
    
  } else {
    rez <- tibble(temp_id = rep(1, n_rows),
                      hits = hits,
                      mentions = mentions)
  }
  rez_meta <- full_join(df_ad, rez, by = "temp_id") %>% 
    select(-temp_id)
  return(rez_meta)
  
}

df_ad <- text_creatives[1, ]
identifier <- cands2$last_name
full_names <- cands2$cand_name
findMentions(df_ad, identifier, full_names)






a <- text_creatives[1:5, ] %>% 
  group_by(ad_id) %>% 
  nest %>% 
  mutate(mentions = map(data, findMentions,
                        identifiers = last_names, 
                        full_names = full_names)) %>% 
  select(-data) %>% 
  unnest
  
all_candidate_mentions <- text_creatives[1:2000, ] %>%
  group_by(ad_id) %>%
  nest %>%
  mutate(mentions = map(data, findMentions,
                        identifiers = last_names,
                        full_names = full_names)) %>%
  select(-data) %>%
  unnest
dim(all_candidate_mentions)

write.csv(all_candidate_mentions, "./output/candidate_mentions_regex2000.csv")
