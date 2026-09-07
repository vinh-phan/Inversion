import polars as pl


#####################################################
################## Aggregating the dataset 
def agg_data(df, type):
    if type == "weather":
        # Create hour/day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.day().alias("day"),           # day of a month 
            pl.col("time").dt.weekday().alias("weekday"),   
            pl.col("time").dt.quarter().alias("quarter"), 
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.year().alias("year")
        )

        # Separate hours into 2AM vs every 6 hours 
        df = df.with_columns([
            pl.when(
                (pl.col("hour") == 2)
            )
            .then(pl.lit("hour_2am"))
            .otherwise(pl.lit("hour_every6"))
            .alias("hour_group")
        ])

        # Truncate inversion strength at 0 
        df = df.with_columns(
            pl.col("t_dif").clip(lower_bound=0).alias("t_dif_truncated")
        )

        # Get squared and cubed terms for t2m (other weather variables already have squares and cubes)
        df = df.with_columns([
            (pl.col("t2m") ** 2).alias("t2m_squared"),
            (pl.col("t2m") ** 3).alias("t2m_cubed")
        ])

        #========= Aggregate data to the daily level =========
        print("Aggregating to the daily level...") 
        variables = [
            't_dif', 't_dif_truncated', 't2m', 'tp', 'u10', 'v10', 'tcc', 'd2m', 'ws', 'wd', 
            't2m_squared', 't2m_cubed', 'tp_squared', 'tcc_squared','d2m_squared',
            'ws_squared', 'wd_squared', 'tp_cubed', 'tcc_cubed', 'd2m_cubed', 'ws_cubed', 
            'wd_cubed'
        ]
        agg_exprs_daily = [
            pl.col("t_dif_indicator").max().alias("t_dif_day"),
            pl.when(pl.col("t_dif_indicator").count()>0).then(pl.sum("t_dif_indicator")).alias("t_dif_n"),
        ] + [pl.col(var).mean().alias(f"{var}_mean") for var in variables]

        df_agg = df.group_by(["lon", "lat", "hour_group", "day", "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)
        
        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("lon", "lat", "year", "month", "day", "hour_group") 

        # Defining aggregation methods for inversion occurrence 
        inv_vars_and_aggs = {
            "t_dif_n": ["rolling_sum", "rolling_mean"],
            "t_dif_day": ["rolling_sum", "rolling_mean"],
        }
        
        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=7)
                .over("lon", "lat", "hour_group")
                .alias(f"{var}_7day_mean")
            )

        # Expressions for rolling means of inversion occurrence 
        for col, aggs in inv_vars_and_aggs.items():
            for agg in aggs:
                rolling_method = getattr(pl.col(col), agg)
                rolling_exprs.append(
                    rolling_method(window_size=7)
                    .over("lon", "lat", "hour_group")
                    .alias(f"{col}_7day_{agg[8:]}")
                )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("lon", "lat", "year", "month", "day", "hour_group") 

        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=30)
                .over("lon", "lat", "hour_group")
                .alias(f"{var}_30day_mean")
            )

        # Expressions for rolling means of inversion occurrence 
        for col, aggs in inv_vars_and_aggs.items():
            for agg in aggs:
                rolling_method = getattr(pl.col(col), agg)
                rolling_exprs.append(
                    rolling_method(window_size=30)
                    .over("lon", "lat", "hour_group")
                    .alias(f"{col}_30day_{agg[8:]}")
                )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)
        
        # Just a note: by construction, t_dif_n and t_dif_day are the same for 2AM
        
        # Create day-month-year-hour variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("hour_group").cast(pl.Utf8)
            ).alias("year_month_day_hour")
        )

        # Create day-month-year (to merge with UKB data later)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        # Return the data 
        df_agg = df_agg.collect(engine = "streaming")

        print("Datasets aggregated.")
        print(f"Unique geographical locations (lon and lat): {df_agg.select(["lon", "lat"]).unique().height}") 
        
        return df_agg
        
    elif type == "pollution_EAC4":
        # Change units of variables 
        df = df.with_columns(
            (pl.col("pm1") * 1000000000).alias("pm1"),
            (pl.col("pm2p5") * 1000000000).alias("pm2p5"),
            (pl.col("pm10") * 1000000000).alias("pm10") 
        )

        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.day().alias("day"),
            pl.col("time").dt.weekday().alias("weekday"),
            pl.col("time").dt.quarter().alias("quarter"),   
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.year().alias("year")
        )

        #========= Aggregate data to the daily level =========
        # Only the PM variables for now - have to check whether it makes sense to look at the other variables, which are in "Total Column")
        print("Aggregating to the daily level...") 
        variables = ['pm1', 'pm2p5', 'pm10' ]
        
        agg_exprs_daily = [pl.col(var).mean().alias(f"{var}_mean") for var in variables]

        df_agg = df.group_by(["lon", "lat", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)

        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("lon", "lat", "year", "month", "day") 
        
        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=7)
                .over("lon", "lat")
                .alias(f"{var}_7day_mean")
            )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("lon", "lat", "year", "month", "day") 

        # Expression for rolling means of weather controls and inversion strengths
        rolling_exprs = []
        for var in variables:
            rolling_exprs.append(
                pl.col(f"{var}_mean")
                .rolling_mean(window_size=30)
                .over("lon", "lat")
                .alias(f"{var}_30day_mean")
            )

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(rolling_exprs)
                
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        # Return the data 
        df_agg = df_agg.collect(engine = "streaming")

        print("Datasets aggregated.")
        print(f"Unique geographical locations (lon and lat): {df_agg.select(["lon", "lat"]).unique().height}") 
        
        return df_agg
    
    elif type == "pollution_NO2":
        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.day().alias("day"),
            pl.col("time").dt.weekday().alias("weekday"),
            pl.col("time").dt.quarter().alias("quarter"),   
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.year().alias("year")
        )
        
        #========= Aggregate data to the daily level =========
        print("Aggregating to the daily level...") 

        agg_exprs_daily = [pl.col("no2").mean().alias("no2_mean")]

        df_agg = df.group_by(["site_name", "lat", "lon", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)
     
        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("site_name", "year", "month", "day") 
        
        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("no2_mean")
                .rolling_mean(window_size=7)
                .over("site_name")
                .alias(f"no2_7day_mean"))

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("site_name", "year", "month", "day") 

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("no2_mean")
                .rolling_mean(window_size=30)
                .over("site_name")
                .alias("no2_30day_mean"))
            
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        print("Datasets aggregated.")
        return df_agg
        
    elif type == "DEFRA_PM2.5":
        # Create day/weekday/month/year indicator
        df = df.with_columns(
            pl.col("date").dt.hour().alias("hour"),
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.quarter().alias("quarter"),   
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year")
        )
        
        #========= Aggregate data to the daily level =========
        print("Aggregating to the daily level...") 

        agg_exprs_daily = [pl.col("pollution_value").mean().alias("DEFRA_pm2p5_mean")]

        df_agg = df.group_by(["site_name", "latitude", "longitude", "day",  "weekday", "month", "quarter", "year"]).agg(agg_exprs_daily)
     
        #========= Aggregate data to the weekly level =========
        print("Aggregating to the weekly level...")

        df_agg = df_agg.sort("site_name", "year", "month", "day") 
        
        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("DEFRA_pm2p5_mean")
                .rolling_mean(window_size=7)
                .over("site_name")
                .alias(f"DEFRA_pm2p5_7day_mean"))

        #========= Aggregate data to the monthly level (30 days) =========
        print("Aggregating to the monthly level...")
        
        df_agg = df_agg.sort("site_name", "year", "month", "day") 

        # Create rolling variables (aggregation)
        df_agg = df_agg.with_columns(pl.col("DEFRA_pm2p5_mean")
                .rolling_mean(window_size=30)
                .over("site_name")
                .alias("DEFRA_pm2p5_30day_mean"))
            
        # Create day-month-year variable (so that each lat-lon is unique when grouping by this variable)
        df_agg = df_agg.with_columns((
            pl.col("year").cast(pl.Utf8) +
            "-" +
            pl.col("month").cast(pl.Utf8).str.zfill(2) + 
            "-" + 
            pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("year_month_day")
        )

        print("Datasets aggregated.")
        return df_agg
        