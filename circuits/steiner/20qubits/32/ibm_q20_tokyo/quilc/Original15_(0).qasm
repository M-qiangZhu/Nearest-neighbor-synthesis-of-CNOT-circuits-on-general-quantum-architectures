// EXPECTED_REWIRING [0 1 2 3 6 5 4 7 8 9 10 13 11 12 14 15 16 17 18 19]
// CURRENT_REWIRING [1 10 8 5 14 11 4 2 3 9 6 7 0 17 15 13 12 16 18 19]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[18];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-2.635223146898034*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.7506271132676074*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.6601985854043217*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.3227130842048*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[3];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[7];
rz(-2.0771658334866547*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-2.951937133979066*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9310081379154876*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5225441851126877*pi) q[3];
rx(1.5707963267948966*pi) q[6];
rz(2.7506271132676146*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[4];
rx(1.5707963267948966*pi) q[5];
cz q[14], q[5];
rz(-1.7843290499389812*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(2.077989633526896*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
rz(-0.8103884456122044*pi) q[16];
cz q[7], q[1];
rz(1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[11], q[9];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-3.075736053375836*pi) q[7];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526896*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.8103884456122044*pi) q[13];
rz(-2.052190394980366*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(0.10344064106915161*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.4189783790674746*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[16], q[14];
rz(1.6366529270088535*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[14];
rx(-1.5707963267948966*pi) q[16];
rz(1.5707963267948966*pi) q[16];
rz(-2.087802470758894*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(1.3844841619731474*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
rz(-2.2762476260936904*pi) q[17];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.075736053375836*pi) q[0];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.3572636036508126*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.063603020062897*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
cz q[13], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[7];
rz(-0.6542456812873576*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9242262418970197*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[5], q[14];
cz q[17], q[16];
rz(-1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(-1.5707963267948966*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
cz q[17], q[16];
rx(-1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[16];
rz(-1.1645820567151592*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(0.16538560610687783*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
rz(-0.5146654427613733*pi) q[17];
rz(1.5707963267948966*pi) q[18];
rx(1.5707963267948966*pi) q[18];
cz q[18], q[19];
rz(-0.057105011344304395*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.18356413639456123*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.606782861698984*pi) q[13];
rz(-0.6542456812873576*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(0.9242262418970197*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
cz q[16], q[14];
rz(1.674236967864048*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4189783790674746*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-3.075736053375836*pi) q[12];
rx(1.5707963267948966*pi) q[18];
cz q[17], q[18];
rz(2.217146941469614*pi) q[16];
rx(-1.5707963267948966*pi) q[16];
rz(1.5707963267948966*pi) q[16];
rz(1.3572636036508117*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(2.077989633526895*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
cz q[17], q[12];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-2.381184772407101*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
cz q[17], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[12];
rz(-0.21353272314407723*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.381184772407101*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[3];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-1.1645820567151595*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.1653856061068779*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.5146654427613733*pi) q[11];
rz(-1.1645820567151592*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(0.16538560610687794*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
rz(2.730367851897572*pi) q[17];
cz q[17], q[16];
rz(1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(1.5707963267948966*pi) q[16];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
cz q[17], q[11];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.432241116400915*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.19598874000363*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.7967559683608076*pi) q[8];
cz q[8], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[1];
cz q[10], q[11];
rx(-1.5707963267948966*pi) q[6];
cz q[12], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[6];
rz(1.550179670184607*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.2678395722928435*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.06900379437231875*pi) q[8];
rz(-1.7843290499389823*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.0636030200628968*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.760407881182692*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[8];
rz(1.5707963267948966*pi) q[19];
rx(1.5707963267948966*pi) q[19];
rz(1.5707963267948966*pi) q[19];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[2], q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
cz q[8], q[7];
rz(0.2427132517316314*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.261599837637769*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[4];
rz(1.9770105968746297*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.9762070474829154*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.6742369678640476*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[12], q[17];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(0.006031654442385636*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.897644314453712*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.7550316154227121*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.6624240052605654*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.17337814656537454*pi) q[12];
cz q[12], q[11];
rz(-0.19576746770944697*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-1.7843290499389806*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.0636030200628972*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7604078811826915*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[2];
rz(0.09301433371850076*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.3981528097392568*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[9];
rz(0.7769136871373916*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674744*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.4688507488946847*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.420521135791635*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.934334098222983*pi) q[12];
cz q[12], q[6];
rz(1.6366529270088543*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-0.9264722246326045*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[6];
rx(1.5707963267948966*pi) q[18];
rz(-1.5707963267948966*pi) q[18];
rz(1.9770105968746403*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.976207047482916*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[18];
rz(0.1404865647060392*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.7362467075559698*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.1011124774600605*pi) q[13];
rz(2.400345941323009*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9966718277252653*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(-1.5707963267948966*pi) q[13];
rz(-1.172567464274844*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[4];
rz(2.025300184227917*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.8197998853773953*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.5309275462226783*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.2426473828505795*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.521939608815274*pi) q[3];
cz q[3], q[2];
rz(0.7174068633636496*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(2.217146941469613*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(1.5707963267948966*pi) q[2];
rz(-1.5280688067575854*pi) q[2];
rz(0.4112248016922271*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.7226142745223185*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.5737567140495635*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.865119934840964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.9676973938157638*pi) q[13];
cz q[13], q[12];
rz(-1.5049397265809397*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-2.0168975849993167*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
cz q[15], q[16];
rz(-2.4952420389150722*pi) q[7];
rx(-1.5707963267948966*pi) q[13];
rz(1.0537901828308993*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.7571084916166462*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[13];
rx(1.5707963267948966*pi) q[13];
rz(-0.7054512992987929*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[13];
rz(-0.6542456812873576*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9242262418970197*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.495242038915076*pi) q[12];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[7];
rz(-1.6384351682257992*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.573868855414932*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.520642090912305*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.068936340237552*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.2961171379881695*pi) q[6];
cz q[6], q[3];
rz(1.3101677170193877*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[3];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[3];
cz q[13], q[12];
rz(3.141592653589793*pi) q[12];
rz(-1.1645820567151592*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687794*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.730367851897572*pi) q[10];
cz q[19], q[10];
rz(1.5707963267948966*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[19], q[18];
rz(1.9770105968746388*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.9762070474829154*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(1.977010596874633*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.9762070474829163*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[9];
rz(-0.12767206220239066*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.31263982029505144*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.2171469414696143*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[6], q[13];
rz(3.141592653589793*pi) q[18];
rx(1.5707963267948966*pi) q[18];
cz q[18], q[12];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rz(-2.6269272108284154*pi) q[1];
rx(3.141592653589793*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[2];
rz(1.1081888165511495*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.57837360989626*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.4827320137677251*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[5];
rz(0.06630437234988262*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(3.141592653589793*pi) q[7];
rz(2.085461769556269*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(3.141592653589793*pi) q[9];
rz(-0.1034406410691524*pi) q[10];
rz(-2.285720596105861*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(3.141592653589793*pi) q[12];
rz(-1.5707963267948966*pi) q[13];
rx(3.141592653589793*pi) q[13];
rz(1.5908680050171664*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.9482789338295112*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.628493249906299*pi) q[14];
rz(-1.1645820567151588*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687789*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.626927210828419*pi) q[15];
rz(1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[16];
rz(-1.5707963267948966*pi) q[16];
rz(-1.5707963267948966*pi) q[17];
rx(-1.5707963267948966*pi) q[18];
rz(1.5707963267948966*pi) q[18];
rz(-1.5707963267948966*pi) q[19];
rx(-1.5707963267948966*pi) q[19];
rz(1.5707963267948966*pi) q[19];
