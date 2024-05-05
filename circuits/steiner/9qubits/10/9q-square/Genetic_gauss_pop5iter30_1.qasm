// Initial wiring: [0 2 1 3 4 5 6 7 8]
// Resulting wiring: [0 8 4 2 1 5 7 6 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[4], q[7];
cx q[1], q[0];
cx q[8], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[8], q[3];
cx q[8], q[3];
cx q[2], q[1];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[4], q[3];
cx q[4], q[7];
