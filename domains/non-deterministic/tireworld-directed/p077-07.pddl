(define (problem tireworld-077-07)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l8 l9 - location)
    (:init (road l0 l13) (road l1 l66) (road l10 l17) (road l10 l33) (road l11 l40) (road l11 l55) (road l12 l6) (road l13 l56) (road l14 l32) (road l14 l61) (road l14 l72) (road l15 l40) (road l15 l75) (road l16 l2) (road l17 l10) (road l17 l27) (road l18 l43) (road l18 l70) (road l19 l1) (road l2 l64) (road l20 l29) (road l21 l55) (road l22 l67) (road l23 l9) (road l24 l59) (road l25 l60) (road l26 l49) (road l26 l68) (road l27 l38) (road l27 l67) (road l28 l33) (road l28 l58) (road l29 l20) (road l3 l49) (road l30 l5) (road l30 l62) (road l31 l19) (road l31 l70) (road l32 l14) (road l32 l4) (road l33 l10) (road l33 l28) (road l34 l46) (road l34 l54) (road l35 l39) (road l35 l60) (road l36 l74) (road l37 l12) (road l38 l19) (road l39 l35) (road l39 l42) (road l4 l32) (road l4 l57) (road l40 l15) (road l41 l50) (road l41 l63) (road l42 l21) (road l42 l22) (road l42 l53) (road l43 l18) (road l44 l3) (road l45 l73) (road l46 l57) (road l47 l44) (road l48 l16) (road l49 l26) (road l5 l30) (road l50 l41) (road l51 l47) (road l52 l50) (road l52 l71) (road l53 l33) (road l53 l42) (road l54 l34) (road l55 l11) (road l55 l21) (road l56 l13) (road l56 l69) (road l57 l4) (road l57 l75) (road l58 l0) (road l58 l28) (road l59 l24) (road l59 l54) (road l6 l36) (road l60 l25) (road l60 l35) (road l61 l71) (road l62 l8) (road l63 l41) (road l63 l76) (road l64 l2) (road l64 l7) (road l65 l5) (road l65 l69) (road l66 l1) (road l66 l45) (road l66 l72) (road l67 l27) (road l68 l24) (road l68 l26) (road l69 l56) (road l69 l65) (road l7 l43) (road l70 l31) (road l71 l52) (road l71 l61) (road l72 l14) (road l73 l45) (road l73 l51) (road l74 l23) (road l75 l48) (road l76 l37) (road l76 l63) (road l8 l12) (road l9 l23) (road l9 l29) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l18) (spare-in l19) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l8) (spare-in l9) (vehicle-at l25))
    (:goal (vehicle-at l20))
)