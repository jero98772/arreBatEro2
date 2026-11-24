// ============================================================================
//  th.v  (UN SOLO ARCHIVO)  --  computador de difusion con N y A en tiempo real
//
//  Contiene:
//    diffusion_computer : nucleo (crecimiento + difusion en punto fijo).
//                         N y A se cargan EN TIEMPO DE EJECUCION (2 <= N <= NMAX)
//    uart_rx / uart_tx  : puerto serie
//    fpga_top           : modulo superior para la FPGA (nucleo + UART + LEDs)
//    diffusion_ref, tb, tb_uart : modelo de referencia y testbenches
//                         (solo con -DSIMULATION)
//
//  ---------------------------- FPGA (Tang Nano 20K) -------------------------
//   top: fpga_top    pines: ver tangnano20k.cst    construir/cargar: ./build.sh
//
//  ---------------------------- COMO FUNCIONA EL NUCLEO -----------------------
//   Un paso (ROUND+COMMIT) es un barrido x = 0..n-1 (n+3 ciclos):
//     t1 = a*x mod n, t2 = a^2*x mod n   (se calculan sumando, sin multiplicar)
//     f1 = flujo por la arista {x, t1}, f2 = flujo por la arista {x, t2}
//     Vn[x] -= f1+f2 ; Vn[t1] += f1 ; Vn[t2] += f2       (dispersion)
//     si active[x]          -> nxt[t1], nxt[t2]
//     si active[t1|t2]      -> nxt[x]
//   COMMIT: V <= Vn ; active <= active | nxt.  Es exactamente el mismo calculo
//   que la version paralela original (diffusion_ref, abajo), y el testbench
//   lo compara paso a paso.  Tras cada paso se hace un barrido "en seco" (sin
//   COMMIT) que calcula done = (nxt & ~active) == 0.
//
//  ---------------------------- PROTOCOLO UART (115200 8N1) ------------------
//   PC/Arduino -> FPGA:
//     'C' n ',' a '\n'  configura (decimal ASCII, ej. "C21,2\n"); responde
//                       "OK\r\n" o "ERR\r\n" (N fuera de 2..NMAX o gcd(A,N)!=1)
//                       y reinicia.  Necesita fin de linea ('\n' o '\r').
//     'R'  reinicia (solo C1 activa, V1 = 1) y pausa
//     'T'  avanza UN paso
//     'G'  corre libre       'P'  pausa
//     'S'  pausa y envia foto BINARIA (para host.py):
//            0xA5, NMAX, n, a, y NBYTES bytes = { v_flat , active , done }
//            little-endian, NBYTES = ceil((NMAX*W + NMAX + 1) / 8)
//              bit 0              done
//              bits 1 .. NMAX     active[0 .. NMAX-1]
//              bits NMAX+1 ...    V[x] en [NMAX+1 + x*W +: W], con signo, 1.0 = 2^FRAC
//     'D'  pausa y envia foto en TEXTO (para el Monitor Serie de Arduino):
//            n=21 a=2 done=1 active=6
//            x=1 v=0.1666
//            ...                 (una linea por celda activa)
//
//  ---------------------------- SIMULACION -----------------------------------
//   iverilog -g2005 -DSIMULATION -s tb -P tb.N=21 -P tb.A=2 -o sim th.v && vvp sim
//   iverilog -g2005 -DSIMULATION -s tb_uart -o sim_uart th.v && vvp sim_uart
// ============================================================================
`timescale 1ns/1ps

module diffusion_computer #(
    parameter NMAX  = 64,   // N maximo (<= 128)
    parameter FRAC  = 16,   // bits fraccionarios: 1.0 = 2^FRAC
    parameter W     = 24,   // ancho del voltaje (con signo)
    parameter SHIFT = 3,    // alpha = 1/2^SHIFT
    parameter N0    = 21,   // configuracion al encender
    parameter A0    = 2
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              clr,      // pulso: reinicia el estado (C1 activa, V1 = 1)
    input  wire              step,     // pulso: un paso (ROUND+COMMIT)
    input  wire              cfg,      // pulso: nueva configuracion (cfg_n, cfg_a)
    input  wire [9:0]        cfg_n,
    input  wire [9:0]        cfg_a,
    output reg               cfg_ack,  // pulso: respuesta a cfg
    output reg               cfg_ok,   //   1 = aceptada, 0 = rechazada
    output wire              busy_o,   // 1 = no acepta comandos todavia
    output reg               done_o,   // 1 = el crecimiento ya se saturo
    output wire [7:0]        n_o,
    output wire [7:0]        a_o,      // A mod N
    output wire [7:0]        order_o,  // popcount(active) = orden de A (si done)
    output wire [NMAX-1:0]   active_o,
    output wire [NMAX*W-1:0] v_flat    // celda x en v_flat[x*W +: W]
);
    localparam signed [W-1:0] ONE = 1 <<< FRAC;
    localparam [7:0] AM0 = A0 % N0;
    localparam [7:0] A20 = ((A0 % N0) * (A0 % N0)) % N0;

    localparam [3:0] S_IDLE   = 4'd0, S_START = 4'd1, S_SWEEP = 4'd2,
                     S_DRAIN  = 4'd3, S_COMMIT = 4'd4, S_INIT = 4'd5,
                     S_MOD    = 4'd6, S_GCD   = 4'd7, S_SQR   = 4'd8;

    reg  [3:0] st;
    reg  [7:0] n, am, a2;          // configuracion activa
    reg  [7:0] x, t1, t2;          // barrido
    reg        dry;                // 1 = barrido en seco (no hace COMMIT)

    wire [NMAX-1:0]   act;         // active
    wire [NMAX-1:0]   nx;          // nxt
    wire [NMAX*W-1:0] vw;          // V
    assign active_o = act;
    assign v_flat   = vw;
    assign n_o      = n;
    assign a_o      = am;
    assign busy_o   = (st != S_IDLE) | step | clr | cfg;

    // ------------------------------------------------------------------
    // Etapa 1 del barrido: leer V[x], V[t1], V[t2] y calcular los flujos
    // ------------------------------------------------------------------
    // d / 2^SHIFT redondeando hacia cero  ( = -((-d) >>> SHIFT) si d < 0 )
    function signed [W-1:0] div_pow2_trunc;
        input signed [W-1:0] d;
        reg   signed [W-1:0] t;
        begin
            t = d + {{(W-SHIFT){1'b0}}, {SHIFT{d[W-1]}}};
            div_pow2_trunc = t >>> SHIFT;
        end
    endfunction

    wire signed [W-1:0] vx  = vw[x  * W +: W];
    wire signed [W-1:0] v1  = vw[t1 * W +: W];
    wire signed [W-1:0] v2  = vw[t2 * W +: W];
    wire                ax  = act[x];
    wire                at1 = act[t1];
    wire                at2 = act[t2];
    wire signed [W-1:0] f1  = (ax & at1) ? div_pow2_trunc(vx - v1) : {W{1'b0}};
    wire signed [W-1:0] f2  = (ax & at2) ? div_pow2_trunc(vx - v2) : {W{1'b0}};

    // siguiente indice: t + paso mod n  (t < n, paso < n)
    wire [8:0] x_n  = x  + 8'd1;
    wire [8:0] t1_s = t1 + am;
    wire [8:0] t2_s = t2 + a2;
    wire [8:0] t1_n = (t1_s >= n) ? t1_s - n : t1_s;
    wire [8:0] t2_n = (t2_s >= n) ? t2_s - n : t2_s;

    // registros de la etapa 1 -> etapa 2
    reg                p_v;
    reg  [7:0]         p_x, p_t1, p_t2;
    reg  signed [W-1:0] p_f1, p_f2;
    reg                p_ax, p_a12;
    wire signed [W-1:0] s12 = p_f1 + p_f2;
    wire signed [W-1:0] ns  = -s12;

    // ------------------------------------------------------------------
    // Etapa 2: cada celda acumula lo que le toca (dispersion) y su bit nxt
    // ------------------------------------------------------------------
    wire init_c   = (st == S_INIT);
    wire start_c  = (st == S_START);
    wire commit_c = (st == S_COMMIT) & ~dry;

    genvar gk;
    generate
        for (gk = 0; gk < NMAX; gk = gk + 1) begin : CELL
            reg signed [W-1:0] v, vn;
            reg                a, nxt;
            wire hx = (p_x  == gk);
            wire h1 = (p_t1 == gk);
            wire h2 = (p_t2 == gk);
            // si la celda es x, t1==x implica f1==0 (y t2==x implica f2==0)
            wire signed [W-1:0] dk = hx        ? ns   :
                                     (h1 & h2) ? s12  :
                                     h1        ? p_f1 :
                                     h2        ? p_f2 : {W{1'b0}};
            always @(posedge clk) begin
                if (init_c) begin
                    v <= (gk == 1) ? ONE : {W{1'b0}};
                    a <= (gk == 1);
                end else if (commit_c) begin
                    v <= vn;
                    a <= a | nxt;
                end
                if (start_c) begin
                    vn  <= v;
                    nxt <= 1'b0;
                end else if (p_v) begin
                    vn  <= vn + dk;
                    nxt <= nxt | (p_ax & (h1 | h2)) | (p_a12 & hx);
                end
            end
            assign vw[gk*W +: W] = v;
            assign act[gk]       = a;
            assign nx[gk]        = nxt;
        end
    endgenerate

    // popcount(active)
    integer p;
    reg [7:0] cnt;
    always @* begin
        cnt = 0;
        for (p = 0; p < NMAX; p = p + 1) cnt = cnt + act[p];
    end
    assign order_o = cnt;

    // ------------------------------------------------------------------
    // Maquina de estados: barridos + configuracion
    // ------------------------------------------------------------------
    reg [9:0] cn, ra, g1, g2, acc;
    reg [7:0] k2;

    always @(posedge clk) begin
        cfg_ack <= 1'b0;
        p_v     <= 1'b0;
        if (rst) begin
            st <= S_INIT;  n <= N0;  am <= AM0;  a2 <= A20;
            done_o <= 1'b0;  cfg_ok <= 1'b0;  dry <= 1'b1;
            x <= 0; t1 <= 0; t2 <= 0;
        end else case (st)
            S_IDLE: begin
                if (cfg) begin
                    cn <= cfg_n;  ra <= cfg_a;
                    if (cfg_n < 2 || cfg_n > NMAX) begin
                        cfg_ack <= 1'b1;  cfg_ok <= 1'b0;
                    end else
                        st <= S_MOD;
                end else if (clr)  st <= S_INIT;
                else if (step) begin dry <= 1'b0; st <= S_START; end
            end

            // ---- un paso: barrido real, COMMIT, luego barrido en seco ----
            S_START: begin
                x <= 0;  t1 <= 0;  t2 <= 0;
                st <= S_SWEEP;
            end
            S_SWEEP: begin
                p_v  <= 1'b1;
                p_x  <= x;     p_t1 <= t1;    p_t2 <= t2;
                p_f1 <= f1;    p_f2 <= f2;
                p_ax <= ax;    p_a12 <= at1 | at2;
                x  <= x_n[7:0];
                t1 <= t1_n[7:0];
                t2 <= t2_n[7:0];
                if (x_n == n) st <= S_DRAIN;
            end
            S_DRAIN:  st <= S_COMMIT;       // la etapa 2 aplica el ultimo x
            S_COMMIT: begin
                if (!dry) begin
                    dry <= 1'b1;  st <= S_START;       // ahora el barrido en seco
                end else begin
                    done_o <= ((nx & ~act) == {NMAX{1'b0}});
                    st <= S_IDLE;
                end
            end
            S_INIT: begin                   // las celdas se inicializan aqui
                dry <= 1'b1;  st <= S_START;
            end

            // ---- configuracion: a mod n, gcd(a, n) == 1, a^2 mod n ----
            S_MOD: begin
                if (ra >= cn) ra <= ra - cn;
                else begin g1 <= cn; g2 <= ra; st <= S_GCD; end
            end
            S_GCD: begin
                if (g2 == 0) begin
                    if (g1 == 1) begin acc <= 0; k2 <= ra[7:0]; st <= S_SQR; end
                    else begin cfg_ack <= 1'b1; cfg_ok <= 1'b0; st <= S_IDLE; end
                end else if (g1 >= g2) g1 <= g1 - g2;
                else begin g1 <= g2; g2 <= g1; end
            end
            S_SQR: begin
                if (k2 == 0) begin
                    n <= cn[7:0];  am <= ra[7:0];  a2 <= acc[7:0];
                    cfg_ack <= 1'b1;  cfg_ok <= 1'b1;
                    st <= S_INIT;
                end else begin
                    acc <= (acc + ra >= cn) ? acc + ra - cn : acc + ra;
                    k2  <= k2 - 8'd1;
                end
            end
            default: st <= S_IDLE;
        endcase
    end

`ifdef SIMULATION
    initial if (NMAX > 128) begin
        $display("ERROR: NMAX=%0d > 128", NMAX); $finish;
    end
`endif
endmodule

// ============================================================================
//  UART receptor  (8N1, muestrea a mitad de cada bit)
// ============================================================================
module uart_rx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115200
)(
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid          // pulso de 1 ciclo cuando llega un byte
);
    localparam CPB = CLK_HZ / BAUD;  // ciclos de reloj por bit

    reg        rx1, rx2, busy;
    reg [15:0] cnt;
    reg [3:0]  bitn;
    initial begin rx1 = 1; rx2 = 1; busy = 0; cnt = 0; bitn = 0; data = 0; valid = 0; end

    always @(posedge clk) begin
        rx1 <= rx;  rx2 <= rx1;      // sincroniza la entrada asincrona
        valid <= 1'b0;
        if (!busy) begin
            if (!rx2) begin          // flanco de bajada = bit de inicio
                busy <= 1; cnt <= CPB/2 - 1; bitn <= 0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 0) begin
                if (rx2) busy <= 0;                  // falso arranque
                else     bitn <= 1;
            end else if (bitn <= 8) begin
                data <= {rx2, data[7:1]};            // LSB primero
                bitn <= bitn + 1;
            end else begin
                busy  <= 0;
                valid <= rx2;                        // bit de parada correcto
            end
        end
    end
endmodule

// ============================================================================
//  UART transmisor
// ============================================================================
module uart_tx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115200
)(
    input  wire       clk,
    input  wire [7:0] data,
    input  wire       start,         // pulso de 1 ciclo (solo si busy = 0)
    output reg        busy,
    output reg        tx
);
    localparam CPB = CLK_HZ / BAUD;

    reg [15:0] cnt;
    reg [3:0]  bitn;
    reg [9:0]  sh;                   // {parada, dato[7:0], inicio}
    initial begin busy = 0; tx = 1; cnt = 0; bitn = 0; sh = 10'h3FF; end

    always @(posedge clk) begin
        if (!busy) begin
            tx <= 1'b1;
            if (start) begin
                sh <= {1'b1, data, 1'b0};
                busy <= 1; cnt <= CPB - 1; bitn <= 0;
                tx <= 1'b0;                          // bit de inicio
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 9) busy <= 0;
            else begin
                tx   <= sh[bitn + 1];
                bitn <= bitn + 1;
            end
        end
    end
endmodule

// ============================================================================
//  fpga_top : nucleo + UART + LEDs.  ESTE es el modulo superior para la FPGA.
// ============================================================================
module fpga_top #(
    parameter NMAX           = 64,
    parameter CLK_HZ         = 27_000_000,   // Tang Nano 20K: 27 MHz
    parameter BAUD           = 115200,
    parameter LED_ACTIVE_LOW = 1             // LEDs de la Tang Nano 20K: activos en bajo
)(
    input  wire clk,
    input  wire uart_rxd,             // pin RX de la FPGA (viene del PC / Arduino)
    output wire uart_txd,             // pin TX de la FPGA (va al PC / Arduino)
    output wire led_done,             // encendido = crecimiento terminado
    output wire led_run               // encendido = corriendo libre
);
    localparam FRAC   = 16;
    localparam W      = 24;
    localparam SHIFT  = 3;
    localparam NB     = NMAX*W + NMAX + 1;
    localparam NBYTES = (NB + 7) / 8;
    localparam HDR    = 4;                    // 0xA5, NMAX, n, a

    // reset de encendido (~16 ciclos)
    reg [3:0] por;
    initial por = 4'd0;
    wire por_busy = ~&por;
    always @(posedge clk) if (por_busy) por <= por + 4'd1;

    // ---------------- nucleo ----------------
    reg        c_clr, c_step, c_cfg;
    reg  [9:0] pn, pa;
    wire       cfg_ack, cfg_ok, core_busy, done;
    wire [7:0] n_cur, a_cur, order;
    wire [NMAX-1:0]   active;
    wire [NMAX*W-1:0] v_flat;

    diffusion_computer #(.NMAX(NMAX), .FRAC(FRAC), .W(W), .SHIFT(SHIFT)) core (
        .clk(clk), .rst(por_busy), .clr(c_clr), .step(c_step),
        .cfg(c_cfg), .cfg_n(pn), .cfg_a(pa), .cfg_ack(cfg_ack), .cfg_ok(cfg_ok),
        .busy_o(core_busy), .done_o(done), .n_o(n_cur), .a_o(a_cur),
        .order_o(order), .active_o(active), .v_flat(v_flat)
    );

    // ---------------- UART ----------------
    wire [7:0] rx_data;
    wire       rx_valid, tx_busy;
    reg  [7:0] tx_data;
    reg        tx_start;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rx(uart_rxd), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .data(tx_data), .start(tx_start), .busy(tx_busy), .tx(uart_txd));

    // ---------------- foto binaria ----------------
    wire [NBYTES*8-1:0] vec = {v_flat, active, done};

    // ---------------- conversion a decimal (3 digitos BCD) ----------------
    function [11:0] bcd3;
        input [7:0] v;
        integer i;
        reg [19:0] s;
        begin
            s = {12'd0, v};
            for (i = 0; i < 8; i = i + 1) begin
                if (s[11:8]  >= 5) s[11:8]  = s[11:8]  + 4'd3;
                if (s[15:12] >= 5) s[15:12] = s[15:12] + 4'd3;
                if (s[19:16] >= 5) s[19:16] = s[19:16] + 4'd3;
                s = s << 1;
            end
            bcd3 = s[19:8];
        end
    endfunction

    // digito pos (0 = centenas) de un BCD de 3 cifras, sin ceros a la izquierda
    // devuelve {saltar, caracter}
    function [8:0] dig;
        input [11:0] b;
        input [1:0]  pos;
        begin
            case (pos)
                2'd0:    dig = {(b[11:8] == 0),                 4'h3, b[11:8]};
                2'd1:    dig = {(b[11:8] == 0 && b[7:4] == 0),  4'h3, b[7:4]};
                default: dig = {1'b0,                           4'h3, b[3:0]};
            endcase
        end
    endfunction

    // ---------------- estado del control ----------------
    localparam [2:0] P_NONE = 0, P_CLR = 1, P_STEP = 2, P_CFG = 3, P_WACK = 4,
                     P_SNAP = 5, P_DUMP = 6;
    localparam [1:0] M_BIN = 0, M_STR = 1, M_DUMP = 2;
    localparam [2:0] D_HDR0 = 0, D_HDR = 1, D_FIND = 2, D_PREP = 3, D_FRAC = 4, D_CELL = 5;

    reg        run, sending, str_err;
    reg [2:0]  pend;
    reg [1:0]  mode;
    reg [15:0] idx;
    reg [1:0]  ps;                    // parser de "C n , a \n"
    reg        pgn, pga, perr;

    // volcado de texto
    reg [2:0]  ds;
    reg [4:0]  lp;                    // posicion en la linea
    reg [7:0]  cx;                    // celda
    reg [11:0] bn, ba, bo, bx, bi;    // BCD
    reg        bd, dneg;
    reg [W-1:0] dv;
    reg [15:0] ff, fdig;
    reg [2:0]  fi;

    initial begin
        c_clr = 0; c_step = 0; c_cfg = 0; pn = 0; pa = 0; tx_data = 0; tx_start = 0;
        run = 0; sending = 0; str_err = 0; pend = P_NONE; mode = M_BIN; idx = 0;
        ps = 0; pgn = 0; pga = 0; perr = 0; ds = 0; lp = 0; cx = 0;
    end

    // byte de la foto binaria
    reg [7:0] bin_byte;
    always @* begin
        case (idx)
            16'd0:   bin_byte = 8'hA5;
            16'd1:   bin_byte = NMAX;
            16'd2:   bin_byte = n_cur;
            16'd3:   bin_byte = a_cur;
            default: bin_byte = vec[8*(idx-HDR) +: 8];
        endcase
    end

    // "OK\r\n" / "ERR\r\n"   ->  {ultimo, caracter}
    reg [8:0] str_ch;
    always @* begin
        if (!str_err) case (idx[2:0])
            3'd0: str_ch = {1'b0, "O"};
            3'd1: str_ch = {1'b0, "K"};
            3'd2: str_ch = {1'b0, 8'h0D};
            default: str_ch = {1'b1, 8'h0A};
        endcase else case (idx[2:0])
            3'd0: str_ch = {1'b0, "E"};
            3'd1: str_ch = {1'b0, "R"};
            3'd2: str_ch = {1'b0, "R"};
            3'd3: str_ch = {1'b0, 8'h0D};
            default: str_ch = {1'b1, 8'h0A};
        endcase
    end

    // linea de cabecera: "n=21 a=2 done=1 active=6\r\n"  -> {ultimo, saltar, char}
    reg [9:0] hdr_ch;
    always @* begin
        case (lp)
            5'd0:  hdr_ch = {2'b00, "n"};
            5'd1:  hdr_ch = {2'b00, "="};
            5'd2:  hdr_ch = {1'b0, dig(bn, 2'd0)};
            5'd3:  hdr_ch = {1'b0, dig(bn, 2'd1)};
            5'd4:  hdr_ch = {1'b0, dig(bn, 2'd2)};
            5'd5:  hdr_ch = {2'b00, " "};
            5'd6:  hdr_ch = {2'b00, "a"};
            5'd7:  hdr_ch = {2'b00, "="};
            5'd8:  hdr_ch = {1'b0, dig(ba, 2'd0)};
            5'd9:  hdr_ch = {1'b0, dig(ba, 2'd1)};
            5'd10: hdr_ch = {1'b0, dig(ba, 2'd2)};
            5'd11: hdr_ch = {2'b00, " "};
            5'd12: hdr_ch = {2'b00, "d"};
            5'd13: hdr_ch = {2'b00, "o"};
            5'd14: hdr_ch = {2'b00, "n"};
            5'd15: hdr_ch = {2'b00, "e"};
            5'd16: hdr_ch = {2'b00, "="};
            5'd17: hdr_ch = {2'b00, bd ? "1" : "0"};
            5'd18: hdr_ch = {2'b00, " "};
            5'd19: hdr_ch = {2'b00, "a"};
            5'd20: hdr_ch = {2'b00, "c"};
            5'd21: hdr_ch = {2'b00, "t"};
            5'd22: hdr_ch = {2'b00, "i"};
            5'd23: hdr_ch = {2'b00, "v"};
            5'd24: hdr_ch = {2'b00, "e"};
            5'd25: hdr_ch = {2'b00, "="};
            5'd26: hdr_ch = {1'b0, dig(bo, 2'd0)};
            5'd27: hdr_ch = {1'b0, dig(bo, 2'd1)};
            5'd28: hdr_ch = {1'b0, dig(bo, 2'd2)};
            5'd29: hdr_ch = {2'b00, 8'h0D};
            default: hdr_ch = {2'b10, 8'h0A};
        endcase
    end

    // linea de celda: "x=1 v=0.1666\r\n"
    reg [9:0] cell_ch;
    always @* begin
        case (lp)
            5'd0:  cell_ch = {2'b00, "x"};
            5'd1:  cell_ch = {2'b00, "="};
            5'd2:  cell_ch = {1'b0, dig(bx, 2'd0)};
            5'd3:  cell_ch = {1'b0, dig(bx, 2'd1)};
            5'd4:  cell_ch = {1'b0, dig(bx, 2'd2)};
            5'd5:  cell_ch = {2'b00, " "};
            5'd6:  cell_ch = {2'b00, "v"};
            5'd7:  cell_ch = {2'b00, "="};
            5'd8:  cell_ch = {1'b0, ~dneg, "-"};
            5'd9:  cell_ch = {1'b0, dig(bi, 2'd0)};
            5'd10: cell_ch = {1'b0, dig(bi, 2'd1)};
            5'd11: cell_ch = {1'b0, dig(bi, 2'd2)};
            5'd12: cell_ch = {2'b00, "."};
            5'd13: cell_ch = {2'b00, 4'h3, fdig[15:12]};
            5'd14: cell_ch = {2'b00, 4'h3, fdig[11:8]};
            5'd15: cell_ch = {2'b00, 4'h3, fdig[7:4]};
            5'd16: cell_ch = {2'b00, 4'h3, fdig[3:0]};
            5'd17: cell_ch = {2'b00, 8'h0D};
            default: cell_ch = {2'b10, 8'h0A};
        endcase
    end
    wire [9:0] line_ch = (ds == D_HDR) ? hdr_ch : cell_ch;

    wire        is_dig = (rx_data >= "0") && (rx_data <= "9");
    wire [3:0]  dval   = rx_data[3:0];
    wire        is_eol = (rx_data == 8'h0A) || (rx_data == 8'h0D);
    wire signed [W-1:0] vcx = v_flat[cx*W +: W];
    wire [19:0] ff10 = {ff, 3'b000} + {ff, 1'b0};      // ff * 10

    always @(posedge clk) begin
        c_clr    <= 1'b0;
        c_step   <= 1'b0;
        c_cfg    <= 1'b0;
        tx_start <= 1'b0;

        // ---------------- comandos recibidos ----------------
        if (rx_valid && !sending && pend == P_NONE) begin
            case (ps)
                2'd0: case (rx_data)
                    "R", "r": begin run <= 1'b0; pend <= P_CLR; end
                    "T", "t": pend <= P_STEP;
                    "G", "g": run  <= 1'b1;
                    "P", "p": run  <= 1'b0;
                    "S", "s": begin run <= 1'b0; pend <= P_SNAP; end
                    "D", "d": begin run <= 1'b0; pend <= P_DUMP; end
                    "C", "c": begin
                        run <= 1'b0; ps <= 2'd1;
                        pn <= 0; pa <= 0; pgn <= 0; pga <= 0; perr <= 0;
                    end
                    default: ;                       // ignora espacios, \r, \n...
                endcase
                2'd1: begin                          // leyendo N
                    if (is_dig) begin
                        if (pn > 10'd99) perr <= 1'b1; else pn <= pn * 10 + dval;
                        pgn <= 1'b1;
                    end else if (rx_data == "," || (rx_data == " " && pgn)) ps <= 2'd2;
                    else if (rx_data == " ") ;
                    else if (is_eol) begin
                        ps <= 2'd0; sending <= 1'b1; mode <= M_STR; str_err <= 1'b1; idx <= 0;
                    end else perr <= 1'b1;
                end
                default: begin                       // leyendo A
                    if (is_dig) begin
                        if (pa > 10'd99) perr <= 1'b1; else pa <= pa * 10 + dval;
                        pga <= 1'b1;
                    end else if (rx_data == " ") ;
                    else if (is_eol) begin
                        ps <= 2'd0;
                        if (perr || !pgn || !pga) begin
                            sending <= 1'b1; mode <= M_STR; str_err <= 1'b1; idx <= 0;
                        end else
                            pend <= P_CFG;
                    end else perr <= 1'b1;
                end
            endcase
        end

        // ---------------- ejecutar lo pendiente cuando el nucleo esta libre ----
        if (!sending && !core_busy) begin
            case (pend)
                P_CLR:  begin c_clr  <= 1'b1; pend <= P_NONE; end
                P_STEP: begin c_step <= 1'b1; pend <= P_NONE; end
                P_CFG:  begin c_cfg  <= 1'b1; pend <= P_WACK; end
                P_SNAP: begin sending <= 1'b1; mode <= M_BIN;  idx <= 0; pend <= P_NONE; end
                P_DUMP: begin sending <= 1'b1; mode <= M_DUMP; ds <= D_HDR0; pend <= P_NONE; end
                P_NONE: if (run) c_step <= 1'b1;
                default: ;
            endcase
        end
        if (pend == P_WACK && cfg_ack) begin
            sending <= 1'b1; mode <= M_STR; str_err <= ~cfg_ok; idx <= 0; pend <= P_NONE;
        end

        // ---------------- transmision ----------------
        if (sending && !tx_busy && !tx_start) begin
            case (mode)
                M_BIN: begin
                    tx_data <= bin_byte; tx_start <= 1'b1;
                    if (idx == HDR + NBYTES - 1) sending <= 1'b0;
                    else                         idx <= idx + 16'd1;
                end
                M_STR: begin
                    tx_data <= str_ch[7:0]; tx_start <= 1'b1;
                    if (str_ch[8]) sending <= 1'b0;
                    else           idx <= idx + 16'd1;
                end
                default: case (ds)                   // M_DUMP
                    D_HDR0: begin
                        bn <= bcd3(n_cur); ba <= bcd3(a_cur); bo <= bcd3(order); bd <= done;
                        lp <= 0; ds <= D_HDR;
                    end
                    D_HDR, D_CELL: begin
                        if (!line_ch[8]) begin tx_data <= line_ch[7:0]; tx_start <= 1'b1; end
                        if (line_ch[9]) begin
                            if (ds == D_HDR) cx <= 0; else cx <= cx + 8'd1;
                            ds <= D_FIND;
                        end else lp <= lp + 5'd1;
                    end
                    D_FIND: begin
                        if (cx >= n_cur)      sending <= 1'b0;   // fin del volcado
                        else if (active[cx]) begin
                            dneg <= vcx[W-1];
                            dv   <= vcx[W-1] ? -vcx : vcx;
                            ds   <= D_PREP;
                        end else cx <= cx + 8'd1;
                    end
                    D_PREP: begin
                        bx <= bcd3(cx);
                        bi <= bcd3(dv[W-1:FRAC]);
                        ff <= dv[FRAC-1:0];
                        fi <= 0;
                        ds <= D_FRAC;
                    end
                    default: begin                   // D_FRAC: 4 decimales
                        fdig <= {fdig[11:0], ff10[19:16]};
                        ff   <= ff10[15:0];
                        fi   <= fi + 3'd1;
                        if (fi == 3'd3) begin lp <= 0; ds <= D_CELL; end
                    end
                endcase
            endcase
        end
    end

    assign led_done = done ^ LED_ACTIVE_LOW[0];
    assign led_run  = run  ^ LED_ACTIVE_LOW[0];
endmodule

`ifdef SIMULATION

// ============================================================================
//  diffusion_ref : el nucleo ORIGINAL (N y A fijos, todo en paralelo).
//  Solo sirve de referencia para comprobar el nucleo nuevo paso a paso.
// ============================================================================
module diffusion_ref #(
    parameter N = 21, parameter A = 2, parameter FRAC = 16,
    parameter W = 24, parameter SHIFT = 3
)(
    input  wire           clk, rst, ce,
    output wire [N-1:0]   active_o,
    output wire           done_o,
    output wire [N*W-1:0] v_flat
);
    function integer modinv;
        input integer a_in; input integer m; integer k;
        begin
            modinv = 0;
            for (k = 1; k < m; k = k + 1) if (((a_in * k) % m) == 1) modinv = k;
        end
    endfunction
    localparam AM = A % N, AINV = modinv(AM, N);
    localparam A2 = (AM * AM) % N, A2INV = (AINV * AINV) % N;
    localparam signed [W-1:0] ONE = 1 <<< FRAC;

    genvar x;
    reg  [N-1:0] active;
    wire [N-1:0] next;
    generate for (x = 0; x < N; x = x + 1) begin : G_NEXT
        assign next[x] = active[(AINV * x) % N] | active[(AM * x) % N]
                       | active[(A2INV * x) % N] | active[(A2 * x) % N];
    end endgenerate
    always @(posedge clk)
        if (rst) active <= ({{(N-1){1'b0}}, 1'b1} << 1);
        else if (ce) active <= active | next;
    assign done_o   = ((next & ~active) == {N{1'b0}});
    assign active_o = active;

    reg  signed [W-1:0] V   [0:N-1];
    wire signed [W-1:0] fA  [0:N-1];
    wire signed [W-1:0] fA2 [0:N-1];
    function signed [W-1:0] div_pow2_trunc;
        input signed [W-1:0] d;
        begin
            if (d < 0) div_pow2_trunc = -((-d) >>> SHIFT);
            else       div_pow2_trunc =   d    >>> SHIFT;
        end
    endfunction
    generate for (x = 0; x < N; x = x + 1) begin : G_FLUX
        localparam TA = (AM * x) % N, TA2 = (A2 * x) % N;
        assign fA[x]  = (active[x] & active[TA])  ? div_pow2_trunc(V[x] - V[TA])  : 0;
        assign fA2[x] = (active[x] & active[TA2]) ? div_pow2_trunc(V[x] - V[TA2]) : 0;
    end endgenerate
    integer k;
    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < N; k = k + 1) V[k] <= 0;
            V[1] <= ONE;
        end else if (ce) begin
            for (k = 0; k < N; k = k + 1)
                V[k] <= V[k] - fA[k] - fA2[k] + fA[(AINV * k) % N] + fA2[(A2INV * k) % N];
        end
    end
    generate for (x = 0; x < N; x = x + 1) begin : G_OUT
        assign v_flat[x*W +: W] = V[x];
    end endgenerate
endmodule

// ============================================================================
//  tb : configura el nucleo por su puerto cfg y lo compara PASO A PASO con
//       el nucleo original (diffusion_ref) para el mismo N y A.
//   iverilog -g2005 -DSIMULATION -s tb -P tb.N=64 -P tb.A=5 -o sim th.v && vvp sim
// ============================================================================
module tb;
    parameter N     = 21;
    parameter A     = 2;
    parameter NMAX  = 64;
    parameter NSTEP = 400;
    localparam FRAC = 16, W = 24, SHIFT = 3;

    reg clk, rst, clr, step, cfg, ref_rst, ref_ce;
    reg [9:0] cfg_n, cfg_a;
    wire cfg_ack, cfg_ok, busy, done;
    wire [7:0] n_o, a_o, order;
    wire [NMAX-1:0]   active;
    wire [NMAX*W-1:0] v_flat;
    wire [N-1:0]      r_active;
    wire              r_done;
    wire [N*W-1:0]    r_v;

    diffusion_computer #(.NMAX(NMAX), .FRAC(FRAC), .W(W), .SHIFT(SHIFT)) dut (
        .clk(clk), .rst(rst), .clr(clr), .step(step), .cfg(cfg),
        .cfg_n(cfg_n), .cfg_a(cfg_a), .cfg_ack(cfg_ack), .cfg_ok(cfg_ok),
        .busy_o(busy), .done_o(done), .n_o(n_o), .a_o(a_o), .order_o(order),
        .active_o(active), .v_flat(v_flat));

    diffusion_ref #(.N(N), .A(A), .FRAC(FRAC), .W(W), .SHIFT(SHIFT)) ref0 (
        .clk(clk), .rst(ref_rst), .ce(ref_ce),
        .active_o(r_active), .done_o(r_done), .v_flat(r_v));

    initial clk = 0;
    always #5 clk = ~clk;

    integer errors, x, s, total, ref_order, t, it;
    reg signed [W-1:0] vx, rv;

    task wait_idle;
        begin @(negedge clk); while (busy) @(negedge clk); end
    endtask

    // aplica una configuracion; ok_exp = 1 si debe aceptarse
    task try_cfg;
        input integer nn, aa, ok_exp;
        begin
            @(negedge clk); cfg_n = nn; cfg_a = aa; cfg = 1;
            @(negedge clk); cfg = 0;
            while (!cfg_ack) @(negedge clk);
            if (cfg_ok !== ok_exp[0]) begin
                $display("FALLO: cfg(N=%0d, A=%0d) -> ok=%0d, esperado %0d", nn, aa, cfg_ok, ok_exp);
                errors = errors + 1;
            end else
                $display("cfg(N=%0d, A=%0d) -> %s", nn, aa, cfg_ok ? "OK" : "ERR");
            wait_idle;
        end
    endtask

    task compare;
        input integer stepno;
        begin
            total = 0;
            for (x = 0; x < NMAX; x = x + 1) begin
                vx = v_flat[x*W +: W];
                total = total + vx;
                if (x < N) begin
                    rv = r_v[x*W +: W];
                    if (vx !== rv || active[x] !== r_active[x]) begin
                        if (errors < 10)
                            $display("  !! paso %0d celda %0d: V=%0d act=%b  ref V=%0d act=%b",
                                     stepno, x, vx, active[x], rv, r_active[x]);
                        errors = errors + 1;
                    end
                end else if (vx !== 0 || active[x] !== 1'b0) begin
                    if (errors < 10) $display("  !! paso %0d celda %0d >= N no es cero", stepno, x);
                    errors = errors + 1;
                end
            end
            if (done !== r_done) begin
                if (errors < 10) $display("  !! paso %0d: done=%b ref=%b", stepno, done, r_done);
                errors = errors + 1;
            end
            if (total !== (1 << FRAC)) begin
                if (errors < 10) $display("  !! paso %0d: carga no conservada %0d", stepno, total);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0; rst = 1; clr = 0; step = 0; cfg = 0; cfg_n = 0; cfg_a = 0;
        ref_rst = 1; ref_ce = 0;
        repeat (3) @(negedge clk);
        rst = 0;
        wait_idle;

        // orden de referencia
        ref_order = 0; t = A % N;
        for (it = 1; it <= N; it = it + 1) begin
            if (t == 1 && ref_order == 0) ref_order = it;
            t = (t * A) % N;
        end

        // configuraciones invalidas: no deben cambiar nada
        try_cfg(1, 1, 0);
        try_cfg(NMAX + 1, 2, 0);
        try_cfg(10, 4, 0);
        try_cfg(N, N, 0);                 // A = 0 mod N
        try_cfg(N, A + 2 * N, 1);         // la valida, con A sin reducir
        if (n_o !== N || a_o !== A % N) begin
            $display("FALLO: n_o=%0d a_o=%0d", n_o, a_o); errors = errors + 1;
        end

        // la referencia arranca a la vez (try_cfg ya hizo el INIT del nucleo)
        @(negedge clk); ref_rst = 1;
        @(negedge clk); ref_rst = 0;
        @(negedge clk);
        compare(0);

        $display("N=%0d  A=%0d  NMAX=%0d  (orden esperado = %0d)", N, A, NMAX, ref_order);
        for (s = 1; s <= NSTEP; s = s + 1) begin
            @(negedge clk); step = 1; ref_ce = 1;
            @(negedge clk); step = 0; ref_ce = 0;
            wait_idle;
            compare(s);
            if (s <= 6 || s % 100 == 0) begin
                vx = v_flat[1*W +: W];
                $display("paso %4d | activas %3d | done %0d | V1 %8.5f",
                         s, order, done, $itor(vx) / $itor(1 << FRAC));
            end
        end

        if (!done || order !== ref_order) begin
            $display("FALLO: done=%0d order=%0d, esperado %0d", done, order, ref_order);
            errors = errors + 1;
        end
        // 'clr' reinicia
        @(negedge clk); clr = 1; @(negedge clk); clr = 0; wait_idle;
        if (order !== 1 || v_flat[1*W +: W] !== (1 << FRAC)) begin
            $display("FALLO: clr no reinicio"); errors = errors + 1;
        end

        $display("----------------------------------------------");
        if (errors == 0) $display("OK: identico al nucleo original en %0d pasos, orden = %0d",
                                  NSTEP, ref_order);
        if (errors == 0) $display("RESULTADO: PASS");
        else             $display("RESULTADO: FAIL (%0d errores)", errors);
        $finish;
    end
endmodule

// ============================================================================
//  tb_uart : simula un PC / Arduino que manda comandos por UART
// ============================================================================
module tb_uart;
    localparam NMAX   = 32;
    localparam W      = 24;
    localparam FRAC   = 16;
    localparam CLK_HZ = 10_000_000;        // reloj lento para simular rapido
    localparam BAUD   = 115200;
    localparam CPB    = CLK_HZ / BAUD;
    localparam NB     = NMAX*W + NMAX + 1;
    localparam NBYTES = (NB + 7) / 8;
    localparam HDR    = 4;

    reg  clk, rxd;
    wire txd, led_done, led_run;
    initial begin clk = 0; rxd = 1; end
    always #50 clk = ~clk;                 // 10 MHz

    fpga_top #(.NMAX(NMAX), .CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk(clk), .uart_rxd(rxd), .uart_txd(txd),
        .led_done(led_done), .led_run(led_run));

    // ---- PC -> FPGA ----
    task send_byte;
        input [7:0] b;
        integer i;
        begin
            rxd = 0;                       repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rxd = b[i];                repeat (CPB) @(posedge clk);
            end
            rxd = 1;                       repeat (2*CPB) @(posedge clk);
        end
    endtask

    task send_dec;
        input integer v;
        begin
            if (v >= 100) send_byte("0" + (v / 100) % 10);
            if (v >= 10)  send_byte("0" + (v / 10) % 10);
            send_byte("0" + v % 10);
        end
    endtask

    // ---- FPGA -> PC ----
    reg [7:0] rxbuf [0:2047];
    integer   rxn, j;
    reg [7:0] rb;
    initial rxn = 0;
    always begin
        @(negedge txd);
        repeat (CPB/2) @(posedge clk);
        repeat (CPB)   @(posedge clk);
        for (j = 0; j < 8; j = j + 1) begin
            rb[j] = txd;                   repeat (CPB) @(posedge clk);
        end
        rxbuf[rxn] = rb;
        rxn = rxn + 1;
    end

    task wait_quiet;
        integer last;
        begin
            last = -1;
            while (last != rxn) begin
                last = rxn;
                repeat (20*CPB) @(posedge clk);
            end
        end
    endtask

    integer errors;

    // "C<n>,<a>\n" y comprueba OK / ERR
    task do_cfg;
        input integer nn, aa, ok_exp;
        reg ok;
        begin
            rxn = 0;
            send_byte("C"); send_dec(nn); send_byte(","); send_dec(aa); send_byte(8'h0A);
            wait_quiet;
            ok = (rxn == 4 && rxbuf[0] == "O" && rxbuf[1] == "K");
            if (!ok && !(rxn == 5 && rxbuf[0] == "E" && rxbuf[1] == "R" && rxbuf[2] == "R")) begin
                $display("  !! respuesta rara a C%0d,%0d (%0d bytes)", nn, aa, rxn);
                errors = errors + 1;
            end else if (ok !== ok_exp[0]) begin
                $display("  !! C%0d,%0d -> %s, esperado %s", nn, aa, ok ? "OK" : "ERR",
                         ok_exp ? "OK" : "ERR");
                errors = errors + 1;
            end else
                $display("C%0d,%0d -> %s", nn, aa, ok ? "OK" : "ERR");
        end
    endtask

    // ---- pide una foto binaria y la decodifica ----
    reg [NBYTES*8-1:0] vec;
    reg signed [W-1:0] vx;
    integer x, i, t, nact, total, tol, fn, fa;
    reg     done_f;

    task snapshot;
        begin
            rxn = 0;
            send_byte("S");
            wait (rxn == HDR + NBYTES);
            repeat (2*CPB) @(posedge clk);
            if (rxbuf[0] !== 8'hA5 || rxbuf[1] !== NMAX) begin
                $display("  !! cabecera incorrecta: %h %h", rxbuf[0], rxbuf[1]);
                errors = errors + 1;
            end
            fn = rxbuf[2]; fa = rxbuf[3];
            for (i = 0; i < NBYTES; i = i + 1) vec[8*i +: 8] = rxbuf[HDR+i];
            done_f = vec[0];
            nact = 0; total = 0;
            for (x = 0; x < NMAX; x = x + 1) begin
                vx = vec[NMAX+1 + x*W +: W];
                total = total + vx;
                if (vec[1+x]) nact = nact + 1;
            end
        end
    endtask

    task check_eq;
        input integer ord;
        begin
            tol = 8 * ord;
            $display("equilibrio N=%0d A=%0d: activas=%0d done=%0d carga=%0d (cada una ~%0d)",
                     fn, fa, nact, done_f, total, (1 << FRAC) / ord);
            if (nact !== ord || done_f !== 1'b1 || total !== (1 << FRAC)) errors = errors + 1;
            for (x = 0; x < NMAX; x = x + 1) begin
                vx = vec[NMAX+1 + x*W +: W];
                if (vec[1+x] && ((vx - (1 << FRAC)/ord > tol) || ((1 << FRAC)/ord - vx > tol))) begin
                    $display("  celda %0d fuera de tolerancia: %0d", x, vx);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        errors = 0;
        repeat (50) @(posedge clk);

        $display("--- configuracion ---");
        do_cfg(10, 4, 0);                  // gcd != 1
        do_cfg(99, 2, 0);                  // N > NMAX
        do_cfg(21, 2, 1);

        $display("--- paso a paso: R, luego T + S ---");
        send_byte("R");
        for (t = 1; t <= 4; t = t + 1) begin
            send_byte("T");
            snapshot;
            vx = vec[NMAX+1 + 1*W +: W];
            $display("paso %0d: activas=%0d done=%0d V1=%8.5f carga=%0d",
                     t, nact, done_f, $itor(vx) / $itor(1 << FRAC), total);
            if (total !== (1 << FRAC)) errors = errors + 1;
            if (t == 2 && (nact !== 6 || done_f !== 1'b1 || vx !== (1 << (FRAC-1)))) begin
                $display("  !! esperaba 6 activas, done=1 y V1=0.5 tras 2 pasos");
                errors = errors + 1;
            end
        end

        $display("--- corrida libre: G, espera, S ---");
        send_byte("G");
        repeat (40000) @(posedge clk);
        snapshot;
        check_eq(6);

        $display("--- volcado de texto (D) ---");
        rxn = 0;
        send_byte("D");
        wait_quiet;
        for (i = 0; i < rxn; i = i + 1) if (rxbuf[i] != 8'h0D) $write("%c", rxbuf[i]);
        if (rxn < 20 || rxbuf[0] !== "n") errors = errors + 1;

        $display("--- otra configuracion: C15,2 (orden 4) ---");
        do_cfg(15, 2, 1);
        send_byte("G");
        repeat (40000) @(posedge clk);
        snapshot;
        check_eq(4);

        if (errors == 0) $display("RESULTADO UART: PASS");
        else             $display("RESULTADO UART: FAIL (%0d errores)", errors);
        $finish;
    end
endmodule
`endif
