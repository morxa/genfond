(define (problem tireworld-072-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l8 l9 - location)
    (:init (road l0 l42) (road l1 l54) (road l10 l58) (road l11 l3) (road l12 l52) (road l13 l16) (road l14 l6) (road l15 l33) (road l16 l35) (road l17 l68) (road l18 l29) (road l18 l55) (road l19 l21) (road l2 l53) (road l20 l69) (road l21 l49) (road l22 l64) (road l23 l37) (road l24 l48) (road l25 l38) (road l26 l20) (road l27 l31) (road l28 l61) (road l29 l18) (road l3 l66) (road l30 l10) (road l31 l46) (road l32 l62) (road l33 l22) (road l34 l3) (road l35 l57) (road l36 l14) (road l37 l15) (road l38 l19) (road l39 l47) (road l4 l29) (road l40 l62) (road l40 l7) (road l41 l50) (road l42 l23) (road l43 l13) (road l44 l65) (road l45 l59) (road l46 l36) (road l47 l17) (road l48 l67) (road l49 l0) (road l5 l63) (road l5 l64) (road l50 l55) (road l51 l53) (road l51 l8) (road l52 l27) (road l52 l45) (road l52 l70) (road l53 l51) (road l54 l25) (road l55 l58) (road l55 l71) (road l56 l26) (road l57 l60) (road l58 l34) (road l59 l41) (road l6 l56) (road l60 l1) (road l61 l24) (road l62 l32) (road l62 l40) (road l63 l4) (road l64 l5) (road l65 l32) (road l66 l9) (road l67 l39) (road l68 l30) (road l69 l43) (road l7 l28) (road l70 l11) (road l70 l44) (road l8 l5) (road l9 l2) (spare-in l0) (spare-in l1) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l30) (spare-in l34) (spare-in l35) (spare-in l38) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l70) (spare-in l8) (vehicle-at l12))
    (:goal (vehicle-at l71))
)