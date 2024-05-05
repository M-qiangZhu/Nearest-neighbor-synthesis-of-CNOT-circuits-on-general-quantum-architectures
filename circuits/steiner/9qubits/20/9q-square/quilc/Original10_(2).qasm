// EXPECTED_REWIRING [0 1 2 8 4 5 6 7 3]
// CURRENT_REWIRING [0 3 1 4 2 6 8 5 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-2.087802470758894*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3844841619731474*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.2762476260936904*pi) q[2];
rz(1.674236967864048*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[8], q[7];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[4];
rz(1.5707963267948966*pi) q[7];
rx(3.141592653589793*pi) q[7];
cz q[7], q[4];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.3572636036508121*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526896*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.381184772407101*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
cz q[7], q[8];
rz(-2.3918013978459243*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-2.9280599304457104*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628976*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.7604078811826911*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.21353272314408495*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[4], q[7];
rz(0.2774272924772014*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.2476173817322804*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.2426515734320869*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.8297426868151945*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.2886931152692072*pi) q[8];
cz q[8], q[3];
rz(2.8770408007532957*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(2.720245155430779*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.721334787752918*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.467355685725745*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.6266113961814117*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.407998081705498*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.9687196414762411*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.1560664449656342*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.8988794018581627*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.879992815952025*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
cz q[3], q[8];
rz(-0.7769136871373913*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7226142745223185*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.9280599304457087*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0779896335268964*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[4];
cz q[4], q[1];
rz(-1.5049397265809397*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[0];
rz(0.24271325173162997*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.2615998376377684*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6734730460682392*pi) q[1];
rz(0.24271325173163086*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.261599837637768*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8973232807266569*pi) q[2];
rz(-1.9216760642735469*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.6951366508133416*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.7435391145662277*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7219669376951143*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.06733832115809665*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(2.0036142183255294*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268964*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8103884456122046*pi) q[6];
cz q[1], q[0];
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
rz(3.0647105556540986*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.401373582907146*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.7857237259107888*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.594474785051473*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.995994364686865*pi) q[4];
rz(0.729113832889174*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.8918587183784752*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(-1.5707963267948966*pi) q[4];
rz(-3.0746142335953177*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[4], q[7];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[4], q[7];
rz(1.2804222416589135*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.6565336180478205*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6791032636096488*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.7859382653076312*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.679801618339631*pi) q[4];
cz q[4], q[3];
rz(-2.753239417940785*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(-1.5707963267948966*pi) q[0];
cz q[1], q[0];
rz(-2.024644567835332*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.3933094723456732*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(-1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.7611842215965474*pi) q[2];
rz(2.4403405833360003*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.7803391768750046*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.738297959803399*pi) q[3];
rz(-2.2853216785276476*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.6463506146747173*pi) q[5];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687805*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.62692721082842*pi) q[6];
rz(-2.6440745164434643*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.035692072123985*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.6586512369843924*pi) q[7];
rz(3.0381520125206407*pi) q[8];
