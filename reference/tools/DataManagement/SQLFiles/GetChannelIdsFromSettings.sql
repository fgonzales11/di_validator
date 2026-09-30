
Select name,
Case
when name like '%Voltage%' then 'voltage'
when name like '%Received%' then 'generation'
when name like '%Delivered%' then 'consumption'
when name like '%kVARh%' then 'reactive'
end as ChannelName, Value from dbo.ViewLookupSettings
where name like 'AMI Analytics - Channels -%Interval%'
and len(value) > 0