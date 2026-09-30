

with src as (
select servicepointid, mtrnum, setdate, removedate, t.deviceid, solar_prob from meterinstallation mi
inner join {destination_table} t on t.deviceid = mi.deviceid and setdate < {startdate} and (removedate > {enddate} or removedate is null)
)
merge into project.SolarSP as trg
using src on src.servicepointid = trg.servicepointid and src.deviceid = trg.deviceid
when matched 
then update set
	num_tests = num_tests + 1,
	solar_prob_agg = solar_prob_agg + src.solar_prob,
	solar_prob_max = iif(solar_prob > solar_prob_max, solar_prob, solar_prob_max),
	solar_prob_min = iif(solar_prob < solar_prob_min, solar_prob, solar_prob_min)

when not matched by target
then insert (servicepointid,deviceid, num_tests, solar_prob_agg, solar_prob_max, solar_prob_min, inception_date, inception_reg, current_rec_reg)
values (servicepointid,deviceid, 1, solar_prob, solar_prob, solar_prob, null, null, null);