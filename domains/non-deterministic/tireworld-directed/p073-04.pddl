(define (problem tireworld-073-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l8 l9 - location)
    (:init (road l0 l3) (road l0 l69) (road l1 l17) (road l1 l23) (road l10 l36) (road l10 l39) (road l11 l2) (road l11 l32) (road l12 l45) (road l12 l64) (road l13 l28) (road l13 l72) (road l14 l15) (road l14 l41) (road l15 l14) (road l15 l58) (road l16 l48) (road l16 l66) (road l17 l1) (road l17 l5) (road l17 l61) (road l17 l72) (road l18 l26) (road l18 l47) (road l19 l22) (road l19 l25) (road l2 l11) (road l2 l20) (road l2 l50) (road l20 l2) (road l20 l42) (road l21 l37) (road l21 l68) (road l22 l19) (road l22 l38) (road l23 l1) (road l23 l41) (road l24 l31) (road l24 l36) (road l25 l19) (road l25 l4) (road l26 l18) (road l26 l29) (road l27 l3) (road l27 l70) (road l28 l13) (road l28 l45) (road l29 l26) (road l29 l56) (road l3 l0) (road l3 l27) (road l3 l34) (road l3 l59) (road l30 l4) (road l30 l53) (road l31 l24) (road l31 l8) (road l32 l11) (road l32 l36) (road l33 l44) (road l33 l7) (road l34 l3) (road l34 l55) (road l35 l51) (road l35 l6) (road l36 l10) (road l36 l24) (road l36 l32) (road l37 l21) (road l37 l63) (road l38 l22) (road l38 l9) (road l39 l10) (road l39 l9) (road l4 l25) (road l4 l30) (road l40 l49) (road l40 l64) (road l41 l14) (road l41 l23) (road l42 l20) (road l42 l54) (road l43 l44) (road l43 l56) (road l44 l33) (road l44 l43) (road l45 l12) (road l45 l28) (road l46 l54) (road l46 l65) (road l47 l18) (road l47 l48) (road l48 l16) (road l48 l47) (road l48 l61) (road l49 l40) (road l49 l66) (road l5 l17) (road l5 l59) (road l50 l2) (road l50 l70) (road l51 l35) (road l51 l69) (road l52 l63) (road l52 l71) (road l53 l30) (road l53 l58) (road l54 l42) (road l54 l46) (road l55 l34) (road l55 l58) (road l55 l60) (road l56 l29) (road l56 l43) (road l57 l6) (road l57 l67) (road l58 l15) (road l58 l53) (road l58 l55) (road l59 l3) (road l59 l5) (road l6 l35) (road l6 l57) (road l60 l55) (road l60 l7) (road l61 l17) (road l61 l48) (road l62 l67) (road l62 l68) (road l63 l37) (road l63 l52) (road l64 l12) (road l64 l40) (road l65 l46) (road l65 l71) (road l66 l16) (road l66 l49) (road l67 l57) (road l67 l62) (road l68 l21) (road l68 l62) (road l69 l0) (road l69 l51) (road l7 l33) (road l7 l60) (road l70 l27) (road l70 l50) (road l71 l52) (road l71 l65) (road l72 l13) (road l72 l17) (road l8 l31) (road l9 l38) (road l9 l39) (spare-in l11) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l38) (spare-in l42) (spare-in l53) (spare-in l55) (spare-in l59) (vehicle-at l8))
    (:goal (vehicle-at l54))
)