// EXPECTED_REWIRING [1 13 3 2 4 5 6 0 8 9 10 11 12 7 14 15 16 19 18 17]
// CURRENT_REWIRING [14 15 4 13 7 12 18 0 6 11 10 8 3 1 16 2 5 19 9 17]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(0.59368010174542*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.958108965734335*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.4741817804854853*pi) q[6];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[12], q[11];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[7];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
rx(1.5707963267948966*pi) q[1];
cz q[8], q[1];
rz(-1.6742369678640472*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(1.0537901828308989*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731476*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.27624762609369*pi) q[7];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[14], q[13];
rz(0.59368010174542*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.958108965734335*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.4741817804854853*pi) q[3];
rz(-1.784329049938982*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.0636030200628972*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.760407881182692*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[6];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.5936801017454187*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.958108965734335*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6015542728903499*pi) q[5];
rz(1.3572636036508121*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526897*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.3811847724071016*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[5];
cz q[7], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[1];
rz(1.3572636036508143*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526895*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-2.3811847724070976*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
cz q[9], q[8];
rz(0.5170061439639975*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7571084916166464*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.8653450274961033*pi) q[11];
rz(2.01918537676344*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.7929976307045448*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.512234620875577*pi) q[7];
rz(-1.1645820567151584*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687764*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.011336675606529*pi) q[14];
cz q[15], q[14];
rz(-0.9771162250494819*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.9581089657343353*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.6015542728903478*pi) q[13];
rz(-1.7843290499389814*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.3811847724071016*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[13];
rz(-1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[17];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[7];
rz(-0.6435078855677967*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.1074841903175616*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.7143998407429075*pi) q[2];
rz(2.487346972302436*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.217366411692773*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[1];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-0.9244457121201783*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(2.487346972302436*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.217366411692774*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.9770105968746328*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.9762070474829154*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.6224394872713374*pi) q[13];
rz(3.141592653589793*pi) q[16];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[13];
rz(3.044583784689532*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1834836878554578*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.540038380699446*pi) q[14];
rz(-0.21353272314408245*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(2.077989633526897*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[14];
rz(1.6366529270088535*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(-2.3811847724070976*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[14];
rz(1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(1.5707963267948966*pi) q[17];
rz(0.10344064106915161*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[11], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[9];
rz(-1.1645820567151588*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(0.16538560610687744*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[17];
rz(0.49023946067626833*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.9581089657343345*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.474181780485485*pi) q[8];
rz(1.977010596874633*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.9762070474829154*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[11], q[17];
rz(1.1379784352642655*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0636030200628972*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7604078811826915*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[2];
rz(-1.5707963267948966*pi) q[17];
rx(3.141592653589793*pi) q[17];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
rz(-1.1645820567151592*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.16538560610687794*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(2.730367851897572*pi) q[12];
cz q[12], q[7];
rz(-1.5707963267948966*pi) q[18];
rx(1.5707963267948966*pi) q[18];
rz(1.5707963267948966*pi) q[18];
rz(-1.4108076218116832*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.9581089657343367*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.4741817804855*pi) q[13];
rz(-1.0792703651674422*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0854370268289926*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.3268234558116156*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.5175135491487435*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(3.097068967614929*pi) q[14];
cz q[14], q[5];
rz(1.197529508383715*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[5];
rz(-0.8425981608894394*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.0636030200628968*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[8];
cz q[12], q[17];
rz(1.4673556857257448*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4317894542157417*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.3422794120067243*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[18], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[18];
cz q[18], q[12];
rz(-2.2357199068687*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5630625415013064*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.5806601225711674*pi) q[14];
cz q[14], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.4735226418313117*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[11], q[17];
rx(-1.5707963267948966*pi) q[18];
rz(-0.13900687257915578*pi) q[18];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[3];
rz(3.141592653589793*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.2285169147881717*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.495242038915076*pi) q[13];
rz(-1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[16];
rz(-3.0381520125206416*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.075736053375836*pi) q[4];
rz(1.9770105968746312*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.9762070474829154*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[12];
cz q[11], q[18];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
cz q[13], q[16];
rz(-0.6971207428145713*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.18348368785545782*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.6674108731043079*pi) q[11];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(-1.6742369678640463*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.4189783790674746*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-3.075736053375836*pi) q[13];
rz(0.8425981608894367*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(2.0779896335268964*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
rz(-0.8103884456122049*pi) q[16];
rz(2.2989944927003503*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.0636030200628968*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.760407881182692*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[4];
cz q[16], q[13];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[13];
rz(-2.6625757902999436*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8385954038498077*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.387104966695441*pi) q[9];
rz(1.3572636036508121*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.0636030200628972*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.7604078811826924*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(1.9770105968746332*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.976207047482916*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.7533556112810593*pi) q[6];
rz(2.487346972302435*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.217366411692774*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[6];
rz(1.3321061582752094*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(-1.674236967864049*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-0.6435078855677967*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.1074841903175616*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.7143998407429075*pi) q[8];
rz(-1.240840698386732*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.6783756905945895*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.6635481537163443*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.34346822863886*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.4141917836883309*pi) q[11];
cz q[11], q[9];
rz(0.9216517081095912*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[9];
cz q[0], q[1];
rz(2.342392873691523*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.929414845437742*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.4360239321629296*pi) q[9];
rz(1.9873447123938954*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.2050118747633558*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.8909348065197754*pi) q[11];
cz q[11], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-0.682754517204029*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(1.1379784352642641*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.063603020062897*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.7604078811826915*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.7605666154642208*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.1802932211337174*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(-1.5707963267948966*pi) q[19];
rx(1.5707963267948966*pi) q[19];
cz q[19], q[18];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[9], q[0];
rx(-1.5707963267948966*pi) q[7];
rz(-1.1897702886693242*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
rz(0.24271325173162997*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.2615998376377684*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.8973232807266575*pi) q[12];
cz q[12], q[11];
cz q[19], q[10];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[19];
cz q[10], q[19];
rz(-1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.6463506146747164*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.6463506146747164*pi) q[3];
rz(0.2427132517316307*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.261599837637768*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4681196075215537*pi) q[4];
rz(-1.4806152732033842*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.396998172501393*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.04407308335161986*pi) q[5];
rx(1.5707963267948966*pi) q[6];
rz(0.39050310566117896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.1034406410691524*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
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
rz(-1.1645820567151588*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(0.16538560610687789*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
rz(2.626927210828419*pi) q[16];
rx(-1.5707963267948966*pi) q[17];
rz(1.5707963267948966*pi) q[17];
rz(-1.5707963267948966*pi) q[19];
