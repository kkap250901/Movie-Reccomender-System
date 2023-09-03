# Returns the top 5 largest films from the input dataframe
def get_top_model(movies_df, target_column, k=5,): 
    return movies_df.nlargest(k, target_column)


# lists out the options
def list_options(ranked): 

    if len(ranked) == 0: 
        print("No films meeting criteria")
    else:
        for i in range(0,len(ranked)): 
            # print(f"{i+1}. {ranked.iloc[i].title[:-6]}")
            print(f"{i+1}. {ranked.iloc[i].title}")


# Filter reccomendations by genres
def filter_by_genres(movies_df, genre_filter):
    if genre_filter == []:
        genre_filter = movies_df.columns

    return movies_df.loc[movies_df[genre_filter].any(1), :]

# Filter the movies by decasde
def filter_by_year(movies_df, year_filter):
    if year_filter == None:
        return movies_df

    return movies_df.loc[(movies_df["year"].astype(int) >= year_filter) & (movies_df["year"].astype(int) < year_filter + 10), :]


def main(movies_df, genres, target_column): 

    # unique_genres # string or smthn idk 


    ranked = get_top_model(movies_df, target_column)
    list_options(ranked)    
    print("\n")
    
    genre_filter = []
    year_gap_duration = None

    while True: 
        try: 
            user_choice_filter = input("What genre choice would u like? To add a decade filter add a date followed with an 's' <year>s: ")
            
            if user_choice_filter == "clear":
                genre_filter = []
                year_gap_duration = None

            elif user_choice_filter not in genres: 
                
                if user_choice_filter[:-1].isnumeric() and user_choice_filter[-1] == "s":
                    year_gap_duration = int(user_choice_filter[:-1])

                else: 
                    print("genre not present")
            
            else: 
                genre_filter.append(user_choice_filter)

            print(year_gap_duration)
            filtered_movies = filter_by_genres(movies_df, genre_filter)
            filtered_movies = filter_by_year(filtered_movies, year_gap_duration)


            ranked = get_top_model(filtered_movies, target_column)
            print(" ".join(genre_filter))
            list_options(ranked)



        except KeyboardInterrupt: 
            print("Exitting System...")
            return
