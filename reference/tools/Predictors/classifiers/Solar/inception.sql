
with cte as (
select deviceid, Valuedate, lag(value) over (partition by deviceid order by valuedate desc) - value as delta
from dbo.ChannelValueDevice where deviceid in ({devices}) and channelid in ({channels})
)
,cte2 as (
select deviceid, valuedate, row_number() over (partition by deviceid order by valuedate asc) as rn
from cte
where delta > {kwh_thresh})

select deviceid, valuedate from cte2
where rn < {d}
order by rn asc
