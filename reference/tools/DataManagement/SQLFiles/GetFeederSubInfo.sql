


select 
mtrnum,DeviceID,mtrk,UsagePoint,Servicepointid,--setdate, removedate,
Isnull(ServiceAddressLatitude, TransformerLatitude) as sp_lat,
Isnull(serviceAddressLongitude, TransformerLongitude) as sp_lon,
iif(ServiceAddressLatitude is null, 'yes', 'no') as missingLoc,

AccountAddress, AccountAddressTown, AccountAddressState, AccountAddressUnit,

CustomerTransformer,
TransformerCapacity,
TransformerLatitude,
TransformerLongitude,
TransformerManufacturer,
TransformerMountingLoc,

CustomerFeeder
 from project.metercertification where customerfeeder in (
'{Feeders}'
)