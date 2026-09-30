select CustomerFeeder as Feeder, CustomerTransformer as Transformer, CustomerTransformerConfiguration as Phases, 
VoltageHighPN as OperatingVoltage, VoltageLowPN as SecondaryVoltage, ServiceType as TransformerType, t.Latitude, t.Longitude
from dbo.Feeder f
inner join dbo.FeederTransformerHistory fth on fth.feederid = f.feederid
inner join dbo.Transformer t on t.transformerid = fth.transformerid
left join dbo.TransformerConfigurationMap tcm on tcm.TransformerConfigurationID = t.TransformerConfiguationHighID
where customerFeeder in ('{feeders}')