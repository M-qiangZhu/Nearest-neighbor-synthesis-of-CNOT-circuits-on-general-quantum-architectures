// EXPECTED_REWIRING [0 4 2 8 7 1 6 5 3]
// CURRENT_REWIRING [5 2 7 8 6 4 1 3 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-2.087802470758894*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3844841619731474*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.2762476260936904*pi) q[6];
cz q[3], q[8];
cz q[5], q[4];
rz(1.674236967864048*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
cz q[2], q[1];
rz(2.5679678771927095*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.6211416866259375*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.2381990962203135*pi) q[3];
rz(2.535104688332209*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4661201043493028*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(-1.5707963267948966*pi) q[3];
rz(-1.5605055191937145*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(-1.5707963267948966*pi) q[6];
rz(1.3572636036508117*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526897*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.3811847724071007*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.164476428540322*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.1834836878554581*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.33410761952961193*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.035956900811219*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.7459125997448404*pi) q[3];
cz q[3], q[2];
rz(-0.9033854536905837*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.1025343675394685*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[1];
rz(-2.6339632987338826*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5516582836681918*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.6042061613228427*pi) q[4];
rz(-3.014589193843571*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.815952971592673*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9739531476843715*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687794*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.730367851897572*pi) q[3];
cz q[8], q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.674236967864049*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.10779068296161931*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(-1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.9770105968746312*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.976207047482916*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.22067814824891466*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.329023692755694*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.958108965734336*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4741817804854938*pi) q[7];
rz(1.674236967864048*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.5170061439639966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.7571084916166442*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-0.7054512992987928*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(-2.5479125518443757*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4741817804854866*pi) q[0];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-2.495242038915076*pi) q[3];
rz(-0.6076103220911448*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.767260907446226*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.5647509535587258*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.2203114383705131*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.27312050432006646*pi) q[8];
cz q[8], q[7];
rz(-2.2523188598250456*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.7760782848027283*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.4384631225456315*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.8863146373780797*pi) q[7];
rz(-2.547912551844372*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.958108965734335*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.474181780485484*pi) q[1];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[7];
rz(-2.928059930445711*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628972*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7604078811826915*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(0.8943935643287382*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.923243545225164*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[4], q[7];
rz(-2.64441124854274*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.7461376013040292*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.19805284612275253*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.137991693403336*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.3743152010524438*pi) q[8];
cz q[8], q[3];
rz(-1.6249371337257799*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(1.5638568704614602*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.0636030200628972*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7604078811826909*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(2.5548782824379366*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.1675999076713408*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.467355685725745*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.1645820567151557*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687789*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747173*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6463506146747164*pi) q[1];
rz(0.6463506146747173*pi) q[2];
rz(-1.8246783522012215*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[4];
rz(1.0561308840335206*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rz(3.141592653589793*pi) q[7];
rz(2.207674832508133*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.3053782315846028*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.41933996090676207*pi) q[8];
