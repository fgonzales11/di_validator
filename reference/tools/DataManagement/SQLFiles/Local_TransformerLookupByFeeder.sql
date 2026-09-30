select 'Primary' as p,
        [Feeder Id] as Feeder,
        [Location Number] as Transformer,
        Phases, 
        [Operating Voltage], 
        '' [Secondary Voltage],
        '' as [Transformer Type],
        Latitude, 
        Longitude  
        from in_and_fac_primary_meters_7_18_2020 where [Feeder Id] in ('{feeders}')
        union
        select 
        'Non-Primary' as p,
        [Feeder Id] as Feeder,        
        [Location Number] as Transformer, 
        Phases, 
        [Operating Voltage], 
        [Secondary Volts], 
        [Transformer Type], 
        [Lat - Mapped], 
        [Long - Mapped]  
        from in_and_fac_transformers_7_18_2020  where  [Feeder Id] in ('{feeders}');
