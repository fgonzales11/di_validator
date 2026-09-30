select 
DeviceID,
ValueDate,
cs.ChannelID,
TimeOffSetInMinutes,
Value01,
Value02,
Value03,
Value04,
Value05,
Value06,
Value07,
Value08,
Value09,
Value10,
Value11,
Value12,
Value13,
Value14,
Value15,
Value16,
Value17,
Value18,
Value19,
Value20,
Value21,
Value22,
Value23,
Value24
from
dbo.channelvaluewidedevice cvwd
inner join dbo.channelstorage cs on cs.channelstorageid = cvwd.channelstorageid
inner join dbo.channel c on c.channelid = cs.channelid
where 
cs.channelid in ({channels}) and
--
deviceid in ({devices}) and valuedate between '{start}' and '{end}'
