// EXPECTED_REWIRING [15 0 4 14 3 5 6 7 8 9 10 11 13 12 1 2]
// CURRENT_REWIRING [15 2 3 14 5 12 6 10 9 8 7 1 0 11 4 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[13];
rz(1.674236967864048*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.784329049938981*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.0636030200628968*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-0.20971076074387893*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.6328979519877201*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.3077566226668766*pi) q[5];
rz(1.6622189597000563*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9715828813520128*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
rz(-2.863915116892896*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(0.19871047107136922*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.4680499270951484*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.3492769467454035*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(0.5936801017454187*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.958108965734335*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6015542728903499*pi) q[1];
rz(-1.7843290499389812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.0779896335268964*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.381184772407101*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.6463506146747164*pi) q[3];
rz(-1.1645820567151632*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.165385606106878*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(1.8158104490134634*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.8972887507914444*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.4346658107781054*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4946594242283744*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.102551252247363*pi) q[5];
cz q[5], q[4];
rz(-3.0057046110165047*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.5684556255922675*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.7571084916166462*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.8653450274961022*pi) q[2];
rz(-0.5124163830597991*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6629206785367268*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.1486774377442237*pi) q[5];
rz(-0.514881786905423*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.870604936385571*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.085678113700319*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(0.1830262608685168*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7488443886087268*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.9671399382271844*pi) q[4];
cz q[3], q[4];
rz(-2.6625757902999436*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8385954038498077*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.387104966695441*pi) q[1];
rz(-2.087802470758894*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3844841619731474*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.2762476260936904*pi) q[14];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-3.075736053375836*pi) q[7];
rz(-0.10969008582078477*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.0427480046911755*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[8], q[9];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(-0.15527697958741274*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3601542165236769*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.4722214007946397*pi) q[9];
rz(0.5170061439639977*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7571084916166462*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.8653450274961032*pi) q[10];
rz(-1.467355685725745*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4189783790674746*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.9280599304457073*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.0636030200628968*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(1.6366529270088535*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.760407881182692*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
cz q[14], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[1];
rz(2.4958141745967546*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.93921413977834*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.235398867171926*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.2979596861329536*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.2709877172042177*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[12];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[13];
cz q[13], q[12];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.8786592586351174*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.4142742199251*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.2745884528461655*pi) q[13];
rz(-2.3309644862678094*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.6397565708247791*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(-1.5707963267948966*pi) q[13];
rz(3.055335532603495*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(0.24271325173163089*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.2615998376377675*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6734730460682401*pi) q[0];
rz(2.2847746162839035*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.045981260140321*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[15];
rz(-1.467355685725745*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[7], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-2.9647870700435135*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8755889665380852*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.0154432503623312*pi) q[2];
rz(0.1758727027136004*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.3158488060654179*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-2.9660294506616527*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-1.8574792809031235*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9740089032045449*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.786814049042679*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.09807971234245*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.8819310644485578*pi) q[13];
cz q[13], q[12];
rz(0.8425347614400538*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-0.7274646997022008*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0000002357280913*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.49482087584837164*pi) q[1];
rz(2.4413232525434805*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.1785807167822242*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[1], q[14];
rx(-1.5707963267948966*pi) q[1];
rz(-1.2411332949485496*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[1], q[14];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[1], q[14];
rz(2.707281278923109*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3369415838066423*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.4637200065889808*pi) q[2];
rz(2.193641066211495*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.3550194548573467*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.28107429943584*pi) q[13];
cz q[13], q[2];
rz(-1.3482621457319723*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.176411732701733*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[2];
rz(2.969496367687594*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.2328717247648346*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.5450961916120916*pi) q[3];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.1083745354054306*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[0], q[15];
rz(-2.294092002564677*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.8446863889293634*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.2735268533843886*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.7352452652312478*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-1.1645820567151595*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.1653856061068779*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.5146654427613733*pi) q[2];
cz q[0], q[15];
rz(-0.924445712120179*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(0.5926407250851512*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8765653616169948*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.8105363934342016*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.224533054615585*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.14740551877429553*pi) q[6];
cz q[6], q[5];
rz(-2.1300501684202775*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
rz(-2.485040152526155*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.619308308373311*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.1645820567151588*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687789*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.626927210828419*pi) q[10];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.21353272314408467*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.077989633526896*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-0.654245681287357*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970196*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.532530017040374*pi) q[3];
cz q[2], q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
cz q[5], q[10];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970196*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[15];
rx(-1.5707963267948966*pi) q[1];
rz(-2.087802470758895*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3844841619731476*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.4361413542909993*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.4564375502462912*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.426995486693993*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.059616298134544*pi) q[2];
rz(0.03728797812529705*pi) q[3];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.2850763214965173*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.9809140097767004*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.4437776502055957*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.4856480852660514*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.5496741467873252*pi) q[12];
cz q[12], q[11];
rz(-1.9306476485911772*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
cz q[3], q[2];
rz(2.022588090432457*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.8746882497510533*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.6706693052227173*pi) q[12];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-0.8654908819636558*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.101821217172781*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(0.05267051292929992*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.18348368785545746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-0.9033854536905904*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6108734163003418*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.6587917123676632*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.7659991211871555*pi) q[14];
rz(1.9770105968746403*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.976207047482916*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[14];
rz(1.0537901828308989*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3844841619731476*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.27624762609369*pi) q[2];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
rz(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.7843290499389812*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268964*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(0.8160080184663325*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.495242038915076*pi) q[9];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.1002759929844377*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.9770105968746392*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.976207047482915*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.0002253027496484*pi) q[10];
cz q[11], q[10];
rz(0.4112248016922271*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7226142745223185*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rz(-1.5049397265809397*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[12], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[3];
rz(1.4564375502462912*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4269954866939927*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.488819971339647*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
cz q[13], q[14];
rz(-2.6625757902999436*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8385954038498077*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.387104966695441*pi) q[1];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[2], q[13];
rz(0.05267051292929693*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.18348368785545804*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.540038380699441*pi) q[4];
rz(1.3572636036508114*pi) q[11];
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
rz(2.6648930705237315*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7902263384554625*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.2222350489574818*pi) q[6];
cz q[6], q[5];
rz(2.9280599304457073*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.0636030200628974*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7604078811826911*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-2.391801397845925*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.1645820567151592*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.16538560610687794*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.730367851897572*pi) q[2];
rz(0.24271325173163064*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.261599837637768*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.244269372863136*pi) q[3];
cz q[2], q[3];
rz(-1.1645820567151601*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610688036*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(3.005931924537328*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.5728138478917195*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.8229725652563733*pi) q[2];
rz(-0.06639170594196546*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.8202581550033272*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[2], q[13];
rx(-1.5707963267948966*pi) q[2];
rz(-2.9736067280443788*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[2], q[13];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[13];
cz q[2], q[13];
rz(1.674236967864048*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.298994492700352*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0779896335268977*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.3811847724070994*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-2.5633149401607613*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.9212183554908346*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.4829545236914872*pi) q[0];
rz(-1.2535094286387376*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6182966111640429*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(0.7144053602842071*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-2.904033348654148*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.8269996475553048*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.344492265502127*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(2.965856411542836*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-2.619399605939641*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.1516004563230757*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-1.1645820567151588*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687744*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[9];
rz(-0.0889540913093459*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.404561088042203*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.442499261417117*pi) q[0];
rz(1.8900489517335632*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(0.6463506146747173*pi) q[2];
rz(-1.1645820567151595*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.1653856061068779*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5146654427613733*pi) q[3];
rz(-1.1645820567151588*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687789*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.626927210828419*pi) q[4];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-0.6470439336682512*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.495242038915076*pi) q[7];
rz(2.626927210828418*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-0.6267019080787648*pi) q[10];
rx(3.141592653589793*pi) q[10];
rz(-1.1645820567151595*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.1653856061068779*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.5146654427613733*pi) q[11];
rz(1.4564375502462918*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.426995486693993*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-3.059616298134544*pi) q[12];
rz(-2.5432348355712135*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.8696168763052492*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.748271213610986*pi) q[13];
rz(3.141592653589793*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
