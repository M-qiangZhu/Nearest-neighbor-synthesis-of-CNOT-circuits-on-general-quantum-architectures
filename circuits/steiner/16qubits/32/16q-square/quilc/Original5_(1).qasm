// EXPECTED_REWIRING [0 1 5 3 4 2 9 6 8 7 10 11 13 12 14 15]
// CURRENT_REWIRING [5 2 11 3 7 1 10 0 4 6 9 13 14 12 15 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.9617246222810056*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.1169664677955562*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-3.0381520125206416*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.21353272314408464*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.3811847724071016*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.4564375502462914*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4269954866939927*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.08197635545524956*pi) q[2];
rz(3.141592653589793*pi) q[3];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-0.11684953538138552*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9847061932198287*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.7692201347043124*pi) q[9];
rz(2.7938757589357204*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.7850522907831803*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[9];
rz(-1.368011358870414*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.961724622281006*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-2.547912551844376*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.958108965734335*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.6015542728903478*pi) q[8];
rz(-3.0188724275335472*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5874869984047466*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.5687377974128363*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.6216912931147656*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[10];
rz(0.10344064106915161*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(0.59368010174542*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.958108965734335*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.4741817804854853*pi) q[5];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
cz q[10], q[11];
rz(-3.0344123880442946*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.2336687369550763*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.3830287542730306*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9449261453810407*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.4773642479036244*pi) q[10];
cz q[10], q[9];
rz(1.7793881966020466*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(0.10072861736596206*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.8327812548453912*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.578516482809089*pi) q[14];
rz(2.019185376763438*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.792997630704545*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.5122346208755784*pi) q[13];
rz(-2.162750317410058*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.407874928791617*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.160676941541042*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(2.4055145381230307*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-2.2062064922268214*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0367302140690553*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.37520239709658565*pi) q[5];
rz(-0.3997027608176919*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.708962281870777*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9281997190768783*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(0.5514662071331253*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.8297634254590072*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.0990728809102257*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7919254147561958*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.5783180723517134*pi) q[6];
cz q[6], q[5];
rz(-2.95691653663558*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.7164998141935497*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.295346365600505*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(2.466724579009034*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.201281817232807*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(0.15183162372609077*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5574260865167322*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.4836383648525833*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.547449259098046*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(2.466888336878423*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.0832479300031679*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.063603020062897*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7604078811826915*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rz(-1.1645820567151592*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.16538560610687794*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.730367851897572*pi) q[1];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[6], q[1];
rz(3.0381520125206416*pi) q[6];
rx(1.5707963267948966*pi) q[9];
rz(-2.1606769415410403*pi) q[9];
rz(-2.8988794018581636*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.8799928159520256*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.053790182830899*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7571084916166453*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.8653450274961024*pi) q[11];
cz q[6], q[9];
rz(1.4950086714708677*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.3300707128261688*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.8331838008225438*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.8350880501703906*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.1003336062112972*pi) q[14];
cz q[14], q[13];
rz(-1.7048267310296*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[4];
rz(0.8973232807266571*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.9343476394717531*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.398380609556001*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(0.10344064106915161*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.190333144335428*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.0867631499704633*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.148910587208931*pi) q[14];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5216365112890728*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[7], q[0];
rz(2.217146941469614*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.976188853450118*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.56352397725335*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.930443772385014*pi) q[13];
rz(-1.1645820567151532*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687753*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[15];
cz q[2], q[3];
rz(-1.581632754232893*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
rz(2.704721612513273*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.4673556857257442*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.075736053375836*pi) q[0];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.3572636036508117*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.0636030200628976*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7604078811826914*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rz(0.8425981608894318*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(-1.2419042693284188*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(0.24271325173162997*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.2615998376377684*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.6734730460682392*pi) q[13];
rz(1.9770105968746345*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9762070474829163*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(1.9770105968746392*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.976207047482915*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.0002253027496484*pi) q[14];
cz q[15], q[14];
rz(2.298994492700354*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0636030200628968*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7604078811826923*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(0.1034406410691524*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.3572636036508114*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.0636030200628972*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(0.7002820151605427*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.6970321717678893*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(0.8271297992717612*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.6885596757449914*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.009281110175170859*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(2.4897477230283993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(-1.1645820567151601*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687802*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.2427132517316307*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.2615998376377684*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.7619092727643946*pi) q[8];
cz q[7], q[8];
rz(1.445813366708306*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.196034952926317*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.8187782187293435*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.006542116277033*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5992280914898533*pi) q[7];
cz q[7], q[6];
rz(1.172610904442828*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.8301850563536206*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.2474334091542145*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.868985233958788*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(2.5148907455110274*pi) q[14];
rx(3.141592653589793*pi) q[14];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-2.176361447400189*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.463275608678869*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2770066615520554*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rx(1.5707963267948966*pi) q[5];
rz(2.0153568086167986*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(2.217146941469614*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[13];
rz(0.7767051408062562*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(0.2427132517316307*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4681196075215537*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6463506146747164*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[4];
rz(0.4368710410765201*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(3.141592653589793*pi) q[5];
rz(-0.1034406410691524*pi) q[6];
rz(1.194711810335642*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rz(-1.5615152166197284*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[10];
rz(1.4564375502462918*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.426995486693993*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.059616298134544*pi) q[11];
rz(3.141592653589793*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];
