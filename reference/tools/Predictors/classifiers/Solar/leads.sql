with cte as (
Select deviceid, Value 
 from dbo.ChannelValueDevice
where valuedate = '{enddate}' and channelid in ( {generation_reg} )
and value > 0)


Select cvd.deviceid
from dbo.ChannelValueDevice cvd
inner join cte t on t.deviceid = cvd.deviceid
where valuedate = '{startdate}' and channelid in ( {generation_reg} )
and Abs(cvd.value - t.value) > {kwh_thresh}