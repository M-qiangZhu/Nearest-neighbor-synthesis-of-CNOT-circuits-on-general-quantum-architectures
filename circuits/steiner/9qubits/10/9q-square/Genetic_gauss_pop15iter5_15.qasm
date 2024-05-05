// Initial wiring: [0 1 8 2 4 5 6 7 3]
// Resulting wiring: [0 1 4 2 6 7 5 8 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[4], q[1];
cx q[4], q[5];
cx q[4], q[5];
cx q[4], q[5];
cx q[3], q[4];
cx q[7], q[8];
cx q[7], q[8];
cx q[7], q[8];
cx q[4], q[5];
cx q[7], q[4];
cx q[7], q[4];
cx q[3], q[8];
cx q[1], q[2];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[6];
cx q[5], q[0];
cx q[1], q[4];
cx q[7], q[8];
