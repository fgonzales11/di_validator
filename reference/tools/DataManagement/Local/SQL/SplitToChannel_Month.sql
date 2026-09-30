
insert or replace into {}
select valuedate as servicepointid, servicepointid as valuedate, value from IntervalData 
where channeldescription = '{}'
and servicepointid >= {} and servicepointid < {}