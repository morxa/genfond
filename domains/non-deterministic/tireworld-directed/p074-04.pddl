(define (problem tireworld-074-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l8 l9 - location)
    (:init (road l0 l30) (road l1 l43) (road l10 l0) (road l11 l21) (road l11 l26) (road l12 l54) (road l12 l6) (road l13 l62) (road l14 l37) (road l14 l73) (road l15 l37) (road l16 l23) (road l16 l47) (road l17 l25) (road l18 l71) (road l19 l65) (road l19 l69) (road l2 l24) (road l2 l27) (road l2 l38) (road l20 l15) (road l21 l11) (road l21 l13) (road l22 l30) (road l22 l5) (road l23 l16) (road l23 l31) (road l24 l2) (road l24 l51) (road l24 l53) (road l25 l17) (road l25 l45) (road l26 l11) (road l27 l2) (road l27 l49) (road l28 l29) (road l29 l28) (road l29 l56) (road l3 l39) (road l30 l0) (road l30 l22) (road l31 l63) (road l32 l58) (road l33 l9) (road l34 l32) (road l35 l57) (road l36 l6) (road l36 l69) (road l37 l14) (road l38 l2) (road l38 l68) (road l39 l3) (road l39 l66) (road l4 l50) (road l40 l60) (road l41 l4) (road l42 l55) (road l42 l9) (road l43 l8) (road l44 l68) (road l45 l44) (road l46 l52) (road l47 l16) (road l47 l46) (road l47 l51) (road l48 l41) (road l49 l27) (road l49 l57) (road l5 l26) (road l50 l1) (road l51 l24) (road l51 l47) (road l52 l61) (road l53 l47) (road l54 l12) (road l54 l48) (road l55 l10) (road l55 l42) (road l56 l29) (road l56 l67) (road l57 l49) (road l58 l32) (road l58 l65) (road l59 l20) (road l6 l12) (road l60 l7) (road l61 l33) (road l61 l52) (road l62 l72) (road l63 l3) (road l63 l31) (road l64 l70) (road l65 l19) (road l65 l58) (road l66 l34) (road l67 l17) (road l67 l56) (road l68 l38) (road l68 l44) (road l69 l19) (road l69 l36) (road l7 l60) (road l7 l71) (road l70 l28) (road l71 l18) (road l72 l40) (road l72 l62) (road l73 l14) (road l73 l64) (road l8 l59) (road l9 l33) (road l9 l42) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l40) (spare-in l42) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l9) (vehicle-at l35))
    (:goal (vehicle-at l18))
)