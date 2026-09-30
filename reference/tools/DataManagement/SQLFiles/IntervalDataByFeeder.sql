select 
Feeder, 
Valuedate, 
AggregatedUsage, 
WeightedVoltage 
from project.View_intervalDataByFeeder where feeder in ('{feeders}')
and valuedate between '{start}' and '{end}'