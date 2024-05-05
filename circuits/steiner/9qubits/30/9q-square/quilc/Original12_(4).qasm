// EXPECTED_REWIRING [0 1 2 3 4 5 6 7 8]
// CURRENT_REWIRING [0 7 6 8 4 5 1 2 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[4];
rz(-2.087802470758894*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731474*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2762476260936904*pi) q[7];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.21353272314408345*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526895*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-2.381184772407102*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(0.10344064106915161*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[7], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(0.2410478866335328*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.3673580598470023*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.5761709847630136*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.2470543234927469*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.8079623644444158*pi) q[7];
cz q[7], q[4];
rz(-2.282940731300654*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[1];
rz(1.3573060172915903*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7433203856716373*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(2.191734919516696*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.0537901828308989*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3844841619731472*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.27624762609369*pi) q[6];
rz(-2.164476428540317*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1834836878554581*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.5400383806994418*pi) q[1];
rz(2.928059930445708*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.077989633526896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.3811847724071016*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.6742369678640474*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
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
rz(-1.1645820567151601*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687814*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[3], q[8];
rz(-1.2640162681799936*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7353797171904177*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.2899173014737308*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.3881601470752565*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.4936905477115704*pi) q[5];
cz q[5], q[4];
rz(-1.587140213563642*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[5];
rz(-1.9873755982923011*pi) q[5];
rz(-1.685155103343501*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7145971668958004*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.4675736461872138*pi) q[6];
rz(-1.771737287466344*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.7151333352528132*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(1.0561308840335242*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
rz(1.0646313500727411*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[8], q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.7164624025470855*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5206521274498792*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.8429680139685244*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.011509308836882*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.236434202289036*pi) q[2];
cz q[2], q[1];
rz(-0.8822549580074313*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rz(-1.9163775417299966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.9581089657343336*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.213532723144084*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268964*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.238207199899194*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.3811847724071016*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-2.5495500016424657*pi) q[6];
rx(3.141592653589793*pi) q[6];
rz(0.2427132517316307*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.261599837637768*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4681196075215537*pi) q[7];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[5];
cz q[6], q[7];
rz(-0.6522325898946715*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.3622404504686387*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(-0.2449433822617828*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.30951813171903886*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.12379380622999217*pi) q[1];
rz(2.7475930826914428*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.11668316348615*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(-1.5707963267948966*pi) q[1];
rz(1.3912838534409797*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-0.6224957371235589*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(3.0061796671964363*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4849603785843473*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[3], q[8];
rx(1.5707963267948966*pi) q[3];
rz(1.4292552353237378*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[3], q[8];
rz(-2.951179943992257*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.7523673096501975*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.9854879190593643*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.6714413224439806*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.9616798408412082*pi) q[4];
cz q[4], q[1];
rz(3.14017049977926*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.8780926509538158*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.6084165194364144*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5377459743182134*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.8370648951818964*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.357198471814702*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.5888510080096918*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.631178050658286*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.5015162477782558*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.1075518669792799*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.360759995282505*pi) q[3];
cz q[3], q[2];
rz(-2.0620089912626067*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.2763336736229054*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[7];
rz(-2.547912551844378*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.958108965734335*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4741817804854893*pi) q[7];
rz(-1.3125881639177257*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.0161476065183086*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[3], q[8];
rz(-3.0381520125206416*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.7843290499389823*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.0636030200628976*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.760407881182692*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1618333128154232*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526896*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.381184772407102*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(0.9167859768246616*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.4189783790674753*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.928059930445708*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628972*pi) q[4];
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
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.6625757902999436*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.8385954038498077*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.387104966695441*pi) q[3];
rz(1.4564375502462923*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4269954866939927*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.6542456812873574*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970201*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.9785146006095449*pi) q[5];
cz q[4], q[5];
rz(1.6527726822501467*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.1645820567151621*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687814*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.9814579837553605*pi) q[7];
cz q[4], q[7];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(-2.9280599304457082*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628968*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[8], q[7];
rz(-0.6326347722890482*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.0133455239950515*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.2527728896223973*pi) q[2];
rz(-0.8292816218348944*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.6911814941195346*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.2706026190111948*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.687720613261201*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.959419829205205*pi) q[8];
cz q[8], q[3];
rz(0.6653070243415984*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(1.977010596874633*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.9762070474829163*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[7];
rz(-0.30059706586258117*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.4049736103739361*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-2.8988794018581627*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.8799928159520265*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-1.3795521813707947*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.1386166302124394*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.123297129946789*pi) q[1];
rz(3.141592653589793*pi) q[2];
rz(1.5481643605977409*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(2.085461769556269*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.23863234086006863*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(0.897323280726658*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-0.3545307729269367*pi) q[7];
rz(0.2602886965298355*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4641996304617764*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.543226826637002*pi) q[8];
