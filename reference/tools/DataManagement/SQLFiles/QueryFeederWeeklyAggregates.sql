
select Feeder,
DatePart(yyyy, Valuedate) as ISOYear,
datepart(isowk, valuedate) as ISOWeek,
Sum(AggregatedUsage)/1000 as MWH,
Max(AggregatedUsage)/1000 as PeakMWH,
iif(Max(AggregatedUsage) > 0, Sum(AggregatedUsage)/(Max(aggregatedUsage)*Count(AggregatedUsage)), null) as LoadFactor,

Avg(WeightedVoltage) AvgNominalVoltage,
Stdev(WeightedVoltage) StdevNominalVoltage

from project.View_intervalDataByFeeder 
where valuedate >= '{date}'
group by Feeder, DatePart(yyyy, Valuedate), datepart(isowk, valuedate)
having sum(aggregatedUsage) > 0

