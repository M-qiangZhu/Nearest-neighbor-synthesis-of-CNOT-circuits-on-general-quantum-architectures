// EXPECTED_REWIRING [8 2 1 3 5 10 6 7 0 14 4 11 9 12 13 15]
// CURRENT_REWIRING [15 0 2 11 5 6 4 1 7 14 9 3 12 13 10 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[9];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[13];
rx(-1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(1.6297992322573909*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.8830290408178292*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(3.0675647514321436*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.79193182548601*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.7820083914404936*pi) q[13];
cz q[13], q[10];
rz(-1.2688073021852597*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(1.5707963267948966*pi) q[9];
rz(0.3616885111994419*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.8342915343438744*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-0.03905601174399113*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.585766771140011*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.4640252662083353*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.222540845986173*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.6058362886561586*pi) q[6];
cz q[6], q[5];
rz(-2.8844684390109148*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(0.59368010174542*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4741817804854853*pi) q[4];
rz(0.01782129264508316*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.360729438532156*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.5952666073029311*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.1645820567151595*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068779*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5146654427613733*pi) q[5];
rz(0.1251846283370569*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[9], q[10];
cz q[5], q[10];
rz(2.873008215923341*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.0426202741611896*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.4114933066546473*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8903979654986411*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.223106082803774*pi) q[6];
cz q[6], q[1];
rz(-2.775214297616193*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-3.0381520125206416*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6340111219669907*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.9198486984899055*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.8171580169426085*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7013756330683947*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.164476428540317*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1834836878554581*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.5400383806994418*pi) q[8];
rz(1.3572636036508123*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.0636030200628972*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.7604078811826922*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[6], q[9];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-2.087802470758894*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.3844841619731474*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.2762476260936904*pi) q[15];
rz(-1.2839596599632976*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.2945442907002325*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.4171554842609926*pi) q[1];
rz(1.467355685725748*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.927142952475931*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-0.6486415852051017*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[10];
rx(1.5707963267948966*pi) q[6];
rz(-0.6486415852051008*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[1], q[6];
rz(-0.9244457121201792*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[9];
rz(1.674236967864048*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.9503495417617461*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731467*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.4361413542909993*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-2.9280599304457082*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.063603020062897*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.3312042079775885*pi) q[10];
rz(-2.3918013978459243*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.4564375502462918*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.426995486693993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.1590703001254985*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(-3.0381520125206416*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-3.075736053375836*pi) q[3];
rz(-2.3918013978459243*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.9280599304457087*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.063603020062897*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.760407881182692*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(2.5479125518443735*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.18348368785545818*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6674108731043074*pi) q[1];
rx(1.5707963267948966*pi) q[7];
rz(1.927142952475932*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.217146941469611*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(-2.3918013978459243*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.213532723144084*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0636030200628959*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7604078811826922*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-0.9244457121201792*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(2.645942530303275*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4405985803610544*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.10227491637357759*pi) q[7];
rz(2.877831384440362*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];
rz(-1.3773977170298077*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.1439816268269525*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.8597653782938854*pi) q[13];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[9], q[6];
rz(-1.8991361121267838*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7575893450117195*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.6919830505880262*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9771202504609156*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.1674442142843615*pi) q[2];
cz q[2], q[1];
rz(-1.3369889835351758*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.2135327231440851*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.8103884456122047*pi) q[5];
rz(0.10344064106915161*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4189783790674746*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-3.075736053375836*pi) q[12];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[14], q[9];
rz(1.4673556857257442*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.075736053375836*pi) q[9];
rz(1.3572636036508117*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.0636030200628976*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.7604078811826914*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.1894996483078608*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.6276010167154373*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.4487148351084405*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.7520063625696116*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.109923282723671*pi) q[14];
cz q[14], q[9];
rz(-0.4000617370342514*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.1769241800369261*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.38560338964057017*pi) q[8];
cz q[15], q[8];
rz(0.9847261377236092*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9107543005713772*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.013746003557095626*pi) q[1];
rz(-0.5306902370261932*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.451992789573568*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(2.0503373532602587*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.7634745303230959*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.9236158671300796*pi) q[14];
rz(2.087802470758894*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.3844841619731463*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(2.760295975102755*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.436141354291*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.1926357210379237*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.6879015074600165*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.996289468559601*pi) q[0];
rz(-0.828287849150597*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.937881422758666*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(-1.5707963267948966*pi) q[0];
rz(0.0036391889148490453*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-1.8877696910081336*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268964*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.1645820567151595*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.1653856061068779*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.5146654427613733*pi) q[6];
rz(-3.0022858922342888*pi) q[9];
rz(1.3627196146900409*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.18348368785545902*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.6674108731043162*pi) q[8];
cz q[9], q[6];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-0.11834082037033511*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.7150973432421317*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.7805277687676426*pi) q[7];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
rz(0.24271325173163064*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.261599837637768*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(2.244269372863136*pi) q[12];
cz q[11], q[12];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-3.0456001897634537*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.883502505462554*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.623250507247441*pi) q[0];
rz(1.2345016248378897*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6935043501820175*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-0.4078095378469815*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
cz q[6], q[7];
rz(-1.784329049938982*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268955*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.381184772407101*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(3.141592653589793*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
rz(-0.9674273493435175*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.230040538929104*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.6934844795002684*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7205587754779659*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.4354388147589681*pi) q[10];
cz q[10], q[9];
rz(1.3147747130054679*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(2.0599667664064163*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.5702722837813324*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[11], q[4];
rz(0.31518779399524616*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(-2.495242038915075*pi) q[13];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.887769691008133*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.0636030200628974*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7604078811826911*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.5707963267948966*pi) q[12];
rz(0.2427132517316307*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.261599837637768*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.4681196075215537*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[10];
cz q[13], q[10];
rz(-0.6435078855677968*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.1074841903175618*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.7143998407429077*pi) q[8];
rz(2.153834895199296*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9560897476169563*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.429210884207785*pi) q[9];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.1645820567151535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687944*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.977010596874637*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.976207047482915*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.9803566792314969*pi) q[14];
cz q[9], q[14];
rz(-1.4266517624124007*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.615325083945457*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.3876684118968231*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0627001732420363*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.23363775989707*pi) q[5];
cz q[5], q[2];
rz(2.2207969093677793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.3636762418732832*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7528028145903027*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.234117437966096*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.7596985567562908*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.5213104049807424*pi) q[9];
cz q[9], q[6];
rz(-2.0382313835779673*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(2.579852581135595*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.2005104739259935*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.893514977688738*pi) q[0];
cz q[7], q[0];
rz(2.8147404883248157*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.2821635449746975*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.7886152938269033*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.2899499319627465*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(0.10344064106915161*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4189783790674746*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.0537901828308984*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.384484161973147*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.6366529270088535*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.4361413542909993*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.9770105968746552*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.9762070474829128*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.6586951617198906*pi) q[9];
rz(-2.036487563265018*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[13];
rz(-2.928059930445709*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.0636030200628952*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.7604078811826951*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.1839481783970909*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.43706538318987453*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.42255675068714943*pi) q[5];
rz(-0.2307639935754271*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.1784280643630183*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.5479125518443713*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.18348368785545854*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.9033854536905868*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
rz(-1.1645820567151592*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687794*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.730367851897572*pi) q[15];
cz q[8], q[15];
rz(-0.6542456812873576*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9242262418970197*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.495242038915076*pi) q[12];
rz(0.24271325173163114*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.2615998376377684*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-2.9513357349047955*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.9581089657343345*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.474181780485482*pi) q[0];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.467355685725745*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(-0.9244457121201797*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(2.9280599304457073*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.3172810855691692*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8392879761367322*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.0629715366536807*pi) q[1];
rz(-0.0003279323762245734*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7624902810902187*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.006922572245623198*pi) q[0];
rz(-0.9083334983576475*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.901981577736059*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-0.8487601753212246*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.977010596874636*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.976207047482915*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-1.537406335683417*pi) q[5];
rz(-0.3008988200860716*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7002902059143357*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.08292882163428837*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.2097136894453713*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(3.0007087975819235*pi) q[7];
cz q[7], q[6];
rz(1.5288218234791149*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(0.9896857720731937*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.1696132333442866*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.0343567166397603*pi) q[0];
rz(-2.6325691994272975*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4835307063875718*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.619397200390683*pi) q[7];
cz q[7], q[0];
rz(2.5291231417221205*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.321897924640335*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.518934372217667*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0641862304623002*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.3490971879290967*pi) q[1];
rz(-2.2552738034273325*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.9158542267252563*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[7], q[8];
rz(1.7184263953106838*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.3604193469254766*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.0687546380398754*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9157362029132876*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.3103973230483203*pi) q[9];
cz q[9], q[6];
rz(0.6521774787385737*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-0.7918706313900006*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.6365087646089543*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.992809439107889*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.5961993719172702*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.2638291272485733*pi) q[2];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(2.238224632650831*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.1292486846528877*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
cz q[13], q[12];
rz(1.90341676765442*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674748*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.5170061439639977*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7571084916166466*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.7054512992987935*pi) q[10];
cz q[10], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(2.761369489712264*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.9641888827222767*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.9438241621069082*pi) q[14];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
rz(2.030736649719053*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526896*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-2.662575790299943*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8385954038498076*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.8203442871083092*pi) q[9];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.8877696910081343*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268964*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122047*pi) q[15];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-1.1645820567151592*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.16538560610687794*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.730367851897572*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[14];
cz q[15], q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(2.487346972302435*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.2173664116927734*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.6542456812873576*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9242262418970197*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-0.5936801017454174*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1834836878554576*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6674108731043056*pi) q[5];
rz(2.6368340108836565*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-0.6542456812873578*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.92422624189702*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-3.0889221406604936*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.1834836878554584*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.003614218325528*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526896*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.3811847724071016*pi) q[14];
cz q[14], q[9];
rz(-0.903385453690591*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(0.9771162250494735*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.18348368785545807*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.540038380699439*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(3.0381520125206416*pi) q[7];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(0.10113059908578832*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.2718181610245407*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.603092832859943*pi) q[10];
rz(-2.4963351136858614*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.29244494174994917*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(-1.5707963267948966*pi) q[10];
rz(-0.8197404232031076*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
cz q[1], q[0];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[7], q[6];
rz(0.7497912557438681*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5823943505303364*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.4048118204350097*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.125933671872665*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.1567863780767436*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.1034406410691524*pi) q[6];
rz(1.4564375502462918*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4269954866939927*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.08197635545524928*pi) q[10];
rz(-0.1034406410691524*pi) q[11];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(2.761369489712264*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9641888827222767*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.9438241621069082*pi) q[9];
rz(2.217146941469614*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-0.09049790853888308*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.14513100676548138*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.8001319237671836*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.587340383034915*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.42008713148230825*pi) q[8];
cz q[8], q[7];
rz(-1.9262597026139794*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.6999253360106783*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.8882053567017527*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
cz q[10], q[11];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[2];
rz(3.141592653589793*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
rz(2.217146941469614*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-0.4436347266817364*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(2.8167524207635486*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.46820235916429526*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.875944786408441*pi) q[8];
rz(-1.1645820567151588*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687789*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.626927210828419*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(3.141592653589793*pi) q[11];
rz(1.5707963267948966*pi) q[12];
rz(-2.7968988654321176*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0829981417926424*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.6491398579650927*pi) q[13];
rz(-1.1645820567151595*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1653856061068779*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];
