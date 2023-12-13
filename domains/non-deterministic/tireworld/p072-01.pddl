(define (problem tireworld-072-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l8 l9 - location)
    (:init (road l0 l16) (road l0 l51) (road l1 l10) (road l1 l13) (road l1 l14) (road l10 l1) (road l10 l5) (road l11 l52) (road l11 l8) (road l12 l23) (road l12 l53) (road l13 l1) (road l13 l17) (road l14 l1) (road l14 l8) (road l15 l27) (road l15 l32) (road l15 l48) (road l16 l0) (road l16 l19) (road l16 l61) (road l17 l13) (road l17 l32) (road l17 l40) (road l18 l28) (road l18 l67) (road l19 l16) (road l19 l27) (road l2 l21) (road l2 l25) (road l20 l25) (road l20 l59) (road l21 l2) (road l21 l70) (road l22 l33) (road l22 l64) (road l23 l12) (road l23 l9) (road l24 l29) (road l24 l59) (road l25 l2) (road l25 l20) (road l26 l40) (road l26 l62) (road l27 l15) (road l27 l19) (road l27 l43) (road l27 l47) (road l28 l18) (road l28 l42) (road l29 l24) (road l29 l69) (road l3 l30) (road l3 l62) (road l30 l3) (road l30 l38) (road l31 l48) (road l32 l15) (road l32 l17) (road l33 l22) (road l33 l68) (road l34 l46) (road l34 l49) (road l35 l6) (road l35 l64) (road l36 l53) (road l36 l69) (road l37 l41) (road l37 l66) (road l38 l30) (road l38 l60) (road l39 l43) (road l39 l55) (road l4 l51) (road l4 l52) (road l40 l17) (road l40 l26) (road l41 l37) (road l41 l70) (road l42 l28) (road l42 l57) (road l43 l27) (road l43 l39) (road l44 l49) (road l44 l56) (road l45 l58) (road l45 l67) (road l46 l34) (road l46 l50) (road l47 l27) (road l47 l9) (road l48 l15) (road l48 l31) (road l48 l52) (road l48 l6) (road l49 l34) (road l49 l44) (road l5 l10) (road l5 l7) (road l50 l46) (road l50 l57) (road l51 l0) (road l51 l4) (road l52 l11) (road l52 l4) (road l52 l48) (road l53 l12) (road l53 l36) (road l53 l56) (road l54 l58) (road l54 l68) (road l55 l39) (road l55 l65) (road l56 l44) (road l56 l53) (road l57 l42) (road l57 l50) (road l58 l45) (road l58 l54) (road l59 l20) (road l59 l24) (road l6 l35) (road l6 l48) (road l60 l38) (road l60 l66) (road l61 l16) (road l61 l63) (road l62 l26) (road l62 l3) (road l63 l61) (road l63 l65) (road l64 l22) (road l64 l35) (road l65 l55) (road l65 l63) (road l66 l37) (road l66 l60) (road l67 l18) (road l67 l45) (road l67 l71) (road l68 l33) (road l68 l54) (road l69 l29) (road l69 l36) (road l7 l5) (road l7 l71) (road l70 l21) (road l70 l41) (road l71 l67) (road l71 l7) (road l8 l11) (road l8 l14) (road l9 l23) (road l9 l47) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l3) (spare-in l30) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l5) (spare-in l50) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l65) (spare-in l67) (spare-in l69) (spare-in l7) (spare-in l8) (vehicle-at l6))
    (:goal (vehicle-at l31))
)