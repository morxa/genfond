(define (problem tireworld-073-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l8 l9 - location)
    (:init (road l0 l40) (road l1 l41) (road l10 l7) (road l11 l46) (road l12 l4) (road l13 l70) (road l14 l69) (road l15 l2) (road l16 l17) (road l17 l15) (road l18 l33) (road l19 l49) (road l2 l61) (road l20 l5) (road l21 l37) (road l22 l50) (road l23 l34) (road l24 l67) (road l25 l67) (road l26 l29) (road l26 l63) (road l27 l53) (road l28 l72) (road l29 l66) (road l3 l12) (road l3 l18) (road l3 l64) (road l30 l57) (road l31 l51) (road l31 l65) (road l32 l38) (road l33 l8) (road l34 l23) (road l34 l44) (road l35 l39) (road l36 l26) (road l38 l22) (road l39 l35) (road l39 l45) (road l4 l21) (road l40 l0) (road l40 l70) (road l41 l14) (road l42 l43) (road l43 l54) (road l44 l55) (road l45 l10) (road l46 l13) (road l47 l24) (road l48 l25) (road l48 l3) (road l49 l65) (road l5 l23) (road l50 l22) (road l50 l47) (road l51 l31) (road l51 l36) (road l52 l20) (road l53 l35) (road l54 l28) (road l54 l43) (road l55 l60) (road l56 l1) (road l57 l11) (road l58 l52) (road l58 l71) (road l59 l58) (road l6 l9) (road l60 l4) (road l61 l27) (road l62 l59) (road l63 l26) (road l63 l30) (road l64 l32) (road l65 l31) (road l66 l56) (road l67 l68) (road l68 l20) (road l69 l42) (road l7 l0) (road l70 l48) (road l71 l68) (road l72 l28) (road l72 l6) (road l8 l16) (road l9 l62) (spare-in l1) (spare-in l10) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l69) (spare-in l72) (spare-in l8) (spare-in l9) (vehicle-at l19))
    (:goal (vehicle-at l37))
)