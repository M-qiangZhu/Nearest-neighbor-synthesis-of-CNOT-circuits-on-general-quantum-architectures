// EXPECTED_REWIRING [0 6 2 3 5 4 1 7 8 10 9 11 12 14 13 15]
// CURRENT_REWIRING [12 15 5 11 10 7 1 0 13 8 4 2 9 3 14 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.3572636036508126*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.063603020062897*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.760407881182692*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-2.087802470758894*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731474*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2762476260936904*pi) q[7];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.3813725356842579*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.216569774501421*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(3.141592653589793*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.9770105968746357*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.9762070474829154*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.9357032737248936*pi) q[9];
cz q[14], q[9];
rz(-1.1403819859983153*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.393476036911137*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.27547591808428634*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.2964070835993753*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.9902818153045585*pi) q[7];
cz q[7], q[6];
rz(1.5047847293922416*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(0.3087760628964751*pi) q[9];
rx(3.141592653589793*pi) q[9];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.075736053375836*pi) q[0];
rz(1.1449529735790496*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3822020274835143*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.013735680257547666*pi) q[7];
rz(1.6567794768363826*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.7226142745223205*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.623566627314278*pi) q[13];
rz(1.3572636036508112*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.0636030200628972*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.118626900733339*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.7604078811826915*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-0.5562539673237912*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4449682035546516*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[9];
rz(1.674236967864048*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(1.674236967864048*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.784329049938982*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.0636030200628972*pi) q[11];
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
cz q[5], q[6];
rz(-0.6542456812873574*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970199*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.4564375502462918*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4269954866939922*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.1390877871968386*pi) q[11];
cz q[10], q[11];
rz(0.24271325173162997*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.2615998376377684*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6734730460682391*pi) q[4];
rz(-2.547912551844373*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.9581089657343345*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.003614218325529*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268964*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(2.238207199899205*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.381184772407101*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1645820567151568*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687775*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5405103479750473*pi) q[14];
cz q[9], q[14];
rz(2.6416558118618862*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.693357364537474*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.4251237669974569*pi) q[10];
rz(0.09229065243936192*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7560496811817228*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rx(-1.5707963267948966*pi) q[10];
rz(2.1627302620513813*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[10], q[11];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(0.24271325173163075*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.2615998376377684*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.5936801017454199*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.9581089657343345*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.4917069903380336*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.6215986415877506*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.1406239171038606*pi) q[10];
cz q[10], q[9];
rz(2.238207199899204*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8677668867642838*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-2.6625757902999436*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8385954038498077*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.387104966695441*pi) q[9];
rz(-1.7584841447253106*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.8103884456122045*pi) q[14];
rz(0.10344064106915161*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.075736053375836*pi) q[4];
rz(-0.8973232807266571*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(0.2692137848848561*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.495242038915076*pi) q[9];
rz(-3.0381520125206416*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.9770105968746319*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.976207047482914*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(3.06668191515989*pi) q[10];
cz q[5], q[10];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.1645820567151615*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687766*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[15];
rz(-2.381649923104057*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.396143885216275*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.0808701721887566*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.413310032325128*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.2868739944704624*pi) q[5];
cz q[5], q[4];
rz(2.1701087049143464*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.39139833732375684*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8398310442410113*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.6800530462595706*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7243570464721305*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.49822628956762*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3844841619731476*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.27624762609369*pi) q[10];
rz(-2.298994492700352*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268955*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.3811847724071007*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.1645820567151624*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687814*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.5306709359666364*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.5070288705127385*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6413571646376379*pi) q[4];
rz(-0.6542456812873583*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970204*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[7];
rz(1.3572636036508117*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268973*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.381184772407101*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(0.5315322499616502*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.51460423284649*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.743173176591991*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.9205282014762615*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.077804025320098*pi) q[15];
cz q[15], q[14];
rz(1.4135986546230281*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.6124318994694697*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.7654208393780556*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(2.2502316365789445*pi) q[12];
rz(-2.1131091891966824*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6143605783023283*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(-1.5707963267948966*pi) q[12];
rz(-2.2378990936532053*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(-2.8768659376163814*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.261155451107592*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.588597834691809*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.3883788159804955*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.4592900174458894*pi) q[12];
cz q[11], q[12];
rz(2.2609207238497033*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.264819435985094*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.7136894152892819*pi) q[0];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[1];
rz(0.24271325173162997*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.2615998376377684*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6734730460682392*pi) q[6];
rz(2.00361421832553*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0779896335268955*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.35344777371036784*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.560443239067526*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.8375457552751105*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.449411024517376*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.1894996483078608*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.6276010167154373*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.4487148351084405*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.7520063625696116*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.109923282723671*pi) q[7];
cz q[7], q[0];
rz(-0.4000617370342514*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.1645820567151561*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068798*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.4578558403425952*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.188880022822631*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(0.49461155953880803*pi) q[15];
cz q[8], q[15];
rz(-0.562887090277428*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.1438377420064163*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.9900127953679869*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5967683774864483*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(0.039883012814732055*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.2322995606698466*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.775437837816944*pi) q[14];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1645820567151592*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.16538560610687794*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.730367851897572*pi) q[13];
cz q[14], q[13];
rz(2.217146941469614*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(0.24271325173163089*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.2615998376377675*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6734730460682401*pi) q[5];
rz(-1.164582056715154*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068783*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(-0.5146654427613786*pi) q[10];
rz(-1.2813375614365432*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-2.551664224513732*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-3.0579658289024856*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
cz q[5], q[2];
cz q[10], q[11];
cz q[13], q[14];
rz(-2.9113378060766344*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.7243615190400374*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.7445820110403591*pi) q[12];
rz(1.2538229625816593*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.0636030200628974*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.6366529270088535*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.7604078811826916*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[14];
rz(-3.0381520125206416*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4189783790674746*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(1.6366529270088535*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(3.141592653589793*pi) q[3];
rz(-0.246232059072675*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.465735228575576*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.466883168758399*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5214991249058423*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.7309851524697398*pi) q[11];
cz q[11], q[4];
rz(1.073777811752528*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(0.08018197808833925*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.417716935106104*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5089562444925944*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.95129274459445*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.608941825575629*pi) q[4];
cz q[4], q[3];
rz(1.1496178735894684*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-0.37514117574556005*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.1280903697130467*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0967993088907266*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rx(1.5707963267948966*pi) q[10];
rz(0.9098424806006342*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rx(-1.5707963267948966*pi) q[6];
rz(-2.160724755870958*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.5597738596053765*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0575200655831756*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.9526679454531314*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.464586081631058*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[1];
rz(1.48716950210759*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(0.24271325173163089*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.2615998376377675*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6734730460682401*pi) q[5];
rz(2.0503373532602587*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.7634745303230959*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.9236158671300796*pi) q[7];
rz(0.8425981608894346*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268955*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.760295975102755*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.381184772407101*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.977010596874633*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.9762070474829154*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5628313021800725*pi) q[10];
cz q[9], q[10];
rz(-2.239786365163532*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.1745458871653814*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.5405299022060548*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.2863296127509496*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(3.0714966706507068*pi) q[11];
cz q[11], q[4];
rz(1.2384146553520647*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.1645820567151595*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.1653856061068779*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.5146654427613733*pi) q[12];
rz(1.0774967449414437*pi) q[10];
rx(3.141592653589793*pi) q[10];
rz(0.6380454113652267*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.710903799503463*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.6739801094243971*pi) q[11];
cz q[11], q[12];
rz(2.02278701041158*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[8];
rz(2.9280599304457082*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.0636030200628979*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151632*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.165385606106878*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[14];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
rz(0.026451942772345572*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.6986947734517543*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
cz q[5], q[2];
rz(-1.1645820567151814*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610688008*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[14];
cz q[10], q[11];
rz(-2.6081202681837854*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0356766487288454*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.7966231359311204*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.125457056601237*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.02480181952484*pi) q[3];
cz q[3], q[2];
rz(1.6745297006842614*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[6];
rz(-0.41122480169219733*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.418978379067474*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.0537901828308993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.3844841619731465*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.4361413542909993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.4564375502462912*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.426995486693993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.059616298134544*pi) q[10];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-0.21616808586345315*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0125498560105277*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(1.121388064661607*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[5], q[2];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.495242038915076*pi) q[7];
rz(2.481736389854418*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.3228124463927986*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.5909340199697737*pi) q[5];
rz(-0.7266344433772852*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.139773271474608*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
rz(2.2631145847790997*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-2.4187204142802257*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8093604490579691*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.7587344542562686*pi) q[6];
rz(-1.776914749866151*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-2.4322411164009154*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.1959887400036298*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.9156330120238823*pi) q[8];
rz(2.2847547070343244*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.57213048877599*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.37385644649370203*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.0471975511965976*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.4392959745202476*pi) q[6];
cz q[6], q[5];
rz(-2.1483420643068527*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.15925195349413085*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.8763275096012164*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rz(-3.124010583262497*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687783*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(2.9782889338932934*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
cz q[10], q[11];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(0.3758308723191299*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5600099422083709*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5665402732894922*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(2.648884060096215*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.7212971579065264*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.878951458442988*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.1234149533583775*pi) q[5];
rz(1.0537901828308973*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.384484161973151*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.436141354291001*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.5684556255922675*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.7571084916166462*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.8653450274961022*pi) q[13];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[10];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(2.242126044289516*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0774163969993413*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8025048201300454*pi) q[2];
rz(2.018501948621702*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.9731993921400364*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-0.995636261343285*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-0.04495979474319506*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.5025945006733585*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.9805979489809986*pi) q[4];
rz(0.6068327607601527*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3573807076534012*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.217146941469614*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[2], q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.9770105968746314*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.9762070474829154*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.761369489712264*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.9641888827222767*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.9438241621069082*pi) q[7];
rz(-2.8313120961106937*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.09980529698211615*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(-1.0668996401401887*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.6380669615720519*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8694508534463701*pi) q[6];
rz(0.4448842556860047*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.503358895134793*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(-1.5707963267948966*pi) q[6];
rz(0.5902185144567622*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[8];
rz(-1.1245139786213656*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.043232369599847*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.06830603216392378*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0292023077765338*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.540531740758303*pi) q[6];
cz q[7], q[6];
rz(1.4434674280329935*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.041787356607678*pi) q[8];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rz(-0.9244457121201792*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.69687872977296*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[5];
cz q[9], q[14];
rz(2.4739499313897495*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-2.953364644558551*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.17734153657258034*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.136434205002057*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.3350923908044392*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.26143941003850263*pi) q[4];
cz q[4], q[3];
rz(1.25148102380939*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.6851551033435015*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7145971668958004*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.117347544912454*pi) q[10];
cz q[5], q[10];
rz(-2.164476428540317*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1834836878554581*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.5400383806994418*pi) q[1];
cz q[6], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.164582056715158*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687736*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[7], q[6];
rz(2.020375594474347*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.4364266550646763*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.4428372339646343*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.7450770238733189*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.6308188928517637*pi) q[14];
cz q[14], q[9];
rz(-1.9225933674457805*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(1.2243791718743324*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.211512358093887*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-0.5146654427613742*pi) q[7];
cz q[7], q[6];
rz(2.848175776811872*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.7571084916166462*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.865345027496103*pi) q[9];
rz(0.04815082604990126*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7891991028288059*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.6378051320474931*pi) q[4];
rz(0.055580042558643974*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7829020504471398*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(-1.5707963267948966*pi) q[4];
rz(-0.2895857665168089*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(3.141592653589793*pi) q[6];
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
rz(-1.1645820567151588*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687789*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.626927210828419*pi) q[9];
rz(-1.6345710356817467*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.6127633269218427*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.23690103008295485*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.2276316364242588*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.2096621052017396*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.20667317028082766*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.1259765739823706*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.4534062041271239*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.228776149472553*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-2.384108434596426*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6156634944585881*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.5306902370261932*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.451992789573568*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.0022858922342888*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.495242038915076*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rz(-0.4706987953756929*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3602274950045583*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5324455950022902*pi) q[3];
rz(-1.7586804646341119*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6843002306526933*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.2859980962558697*pi) q[4];
rz(3.025831976826404*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(0.6463506146747173*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rz(3.141592653589793*pi) q[9];
rz(2.217146941469614*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
rz(2.870697397393259*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(-1.1645820567151592*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.16538560610687794*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.6269272108284194*pi) q[13];
rz(1.2707561474154692*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3007974560117939*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(0.6610107143962852*pi) q[14];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.2605157693157976*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
