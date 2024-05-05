// EXPECTED_REWIRING [1 5 0 3 4 6 2 7 15 9 10 11 12 14 13 8]
// CURRENT_REWIRING [5 9 1 2 12 3 0 7 15 14 10 13 11 8 4 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(0.5936801017454187*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.958108965734335*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6015542728903499*pi) q[1];
rz(-1.7843290499389832*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526897*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.3811847724071025*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[2];
rz(-0.2135327231440851*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.381184772407101*pi) q[3];
cz q[3], q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.7843290499389812*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.077989633526896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.8103884456122044*pi) q[4];
rz(-2.547912551844373*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(2.238207199899204*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.7843290499389812*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8103884456122044*pi) q[6];
rx(-1.5707963267948966*pi) q[3];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.7843290499389812*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.8103884456122044*pi) q[11];
rz(0.59368010174542*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.958108965734335*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.4741817804854853*pi) q[9];
rz(1.674236967864048*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.2135327231440849*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268955*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.381184772407101*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[4];
cz q[11], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.495242038915076*pi) q[2];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
cz q[2], q[5];
rz(0.6128972238194015*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6344029625411813*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.6009734199202523*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3565964972586744*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(3.140764277659598*pi) q[6];
cz q[6], q[1];
rz(-0.040720426712669955*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.467355685725745*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.8965878265811407*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.6600203156989194*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.7911544093092393*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8879816975807362*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(1.4564375502462914*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4269954866939927*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.08197635545524956*pi) q[6];
rz(1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(-2.547912551844372*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.474181780485484*pi) q[4];
rz(2.4873469723024355*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.217366411692774*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[10];
cz q[6], q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
cz q[7], q[8];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-3.075736053375836*pi) q[8];
rz(-1.164582056715159*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687777*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.147971357474203*pi) q[15];
cz q[14], q[15];
rz(-1.0650709941555163*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.8446139963969208*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.7541069778367087*pi) q[4];
rz(-1.9847015669371415*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.3263992993378881*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(-1.5707963267948966*pi) q[4];
rz(3.07213506245027*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.9770105968746357*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.9762070474829154*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(1.3572636036508108*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.0636030200628974*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.760407881182692*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.3572636036508121*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526897*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-2.3811847724071016*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(1.5878450165689724*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.2559257546632338*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.6058722383902864*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.595928541111952*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.4170799931139997*pi) q[11];
cz q[11], q[4];
rz(-0.5513456218523345*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[12];
rz(-1.6008820993112483*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.579764766497735*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
cz q[12], q[13];
rz(-0.593680101745419*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.1834836878554584*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.667410873104307*pi) q[12];
cz q[13], q[10];
rz(0.10344064106915161*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.075736053375836*pi) q[10];
rz(2.3686612551354327*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7571084916166466*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.8653450274961036*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.3572636036508108*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.0636030200628974*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.760407881182692*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-0.29453954701294005*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.943124079103106*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.5259531711072434*pi) q[5];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.997511718668252*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.2196562130584334*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[10], q[5];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1645820567151595*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.1653856061068779*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.5146654427613733*pi) q[10];
rz(-1.1645820567151595*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.1653856061068779*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5146654427613733*pi) q[13];
cz q[10], q[13];
rz(-1.1645820567151595*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.1653856061068779*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.5146654427613733*pi) q[14];
rz(0.24271325173162997*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.2615998376377684*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8973232807266575*pi) q[6];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
cz q[10], q[9];
cz q[13], q[14];
rz(-1.7209767616475171*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.380213516457588*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.28042050059901946*pi) q[2];
rz(-2.4480636344987525*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.1708689172262385*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-1.055313147482992*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(0.5170061439639977*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7571084916166462*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.8653450274961032*pi) q[6];
rz(2.426231001281402*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.192671757395102*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6244161652666439*pi) q[1];
rz(0.8120929617707714*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8115200055566918*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.9427235039096704*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.5014063293821085*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[5];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.567179584906594*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.18348368785545816*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6674108731043078*pi) q[4];
rz(1.9770105968746323*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.976207047482915*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.8599124796614872*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.9793824456967872*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.4288910084805457*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.313893835105808*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.9336314391450846*pi) q[6];
cz q[6], q[5];
rz(-2.744564727379788*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0711646952372316*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.2475862993211435*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.0437962754931651*pi) q[6];
cz q[1], q[6];
rz(0.45421194255115466*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.3414235544836811*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.5234683773539145*pi) q[3];
cz q[2], q[3];
rz(-0.32448976887137093*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5380403495166333*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.452682411195739*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-1.1645820567151592*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687794*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.730367851897572*pi) q[5];
rz(0.40879634819976474*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.2408554372566054*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3831562006967713*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.0354098370305524*pi) q[8];
rz(-2.9477985677694893*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.178786678219279*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8689396292426705*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[8], q[15];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(-1.6712300912836822*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(1.4035649224594278*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.1013259973558442*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.08972162840226434*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.1620550745676832*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.828449301989676*pi) q[11];
cz q[11], q[4];
rz(-0.3896977883571173*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(2.219500540514438*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.1414788984295632*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.29458756564700966*pi) q[11];
rz(-2.7640947867453733*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.754922226413092*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.6970893459204852*pi) q[12];
rz(1.4946640508447693*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.6761412528013473*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5140034513208636*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.3456910616876576*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.5344950405130484*pi) q[2];
cz q[2], q[1];
rz(-2.2049944195891076*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.3961061874289733*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.2301789619859758*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(1.5707963267948966*pi) q[6];
rz(1.9770105968746357*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9762070474829154*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.7303678518975705*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.4189783790674753*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.8706202020583153*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.547036927446937*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.7786320067861212*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0676216375937564*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.137978435264265*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.077989633526896*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.8103884456122045*pi) q[9];
rz(-0.4287245767004059*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9094485931362617*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.5144181158305003*pi) q[10];
rz(0.3857375896844447*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.4090918485878106*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(-1.5707963267948966*pi) q[10];
rz(0.08465560699593588*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
cz q[12], q[11];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-0.1034406410691524*pi) q[5];
rz(-1.674236967864048*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(0.5545625969437338*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.0703833793494268*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.5214742077943315*pi) q[1];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
cz q[6], q[1];
rz(-1.2065760827405647*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.6818451820157955*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.7635011752285006*pi) q[2];
rz(-2.2798147367286816*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.9005855322458356*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-0.37846675705106914*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(0.508948846003816*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.654245681287358*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.9242262418970197*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5912948181415483*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.849048553135589*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.07450033876634542*pi) q[10];
rz(-1.137978435264266*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0779896335268955*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-3.13312232647771*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7071498256475648*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.671005005966473*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747173*pi) q[0];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(0.9737075274125568*pi) q[2];
rx(3.141592653589793*pi) q[2];
rz(-2.1768703574563197*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.5125882128774757*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5439557646428346*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
rz(2.217146941469614*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.1645820567151595*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.1653856061068779*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.5146654427613733*pi) q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(1.467355685725745*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-2.4952420389150767*pi) q[10];
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
rz(-1.288308681204073*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.2975054390829843*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.9024582298844235*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-0.36700478071668163*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.4955403154498508*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.381308591825455*pi) q[15];
