// Initial wiring: [5 1 4 2 3 0 6 7 8]
// Resulting wiring: [4 1 7 2 3 0 6 8 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[5];
cx q[7], q[6];
cx q[8], q[7];
cx q[8], q[7];
cx q[0], q[5];
cx q[0], q[5];
cx q[7], q[4];
cx q[7], q[4];
cx q[7], q[4];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[0], q[5];
cx q[7], q[6];
cx q[4], q[5];
cx q[2], q[1];
cx q[4], q[7];
