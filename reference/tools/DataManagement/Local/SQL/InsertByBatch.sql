INSERT or replace into IntervalData
select * from stage_intervals where ROWID >= '{}' and ROWID <= '{}';