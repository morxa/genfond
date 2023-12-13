(define (problem tireworld-073-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l8 l9 - location)
    (:init (road l0 l35) (road l0 l61) (road l1 l15) (road l1 l17) (road l10 l12) (road l10 l3) (road l11 l22) (road l11 l47) (road l12 l10) (road l12 l17) (road l13 l15) (road l13 l5) (road l14 l58) (road l14 l6) (road l15 l1) (road l15 l13) (road l15 l56) (road l16 l17) (road l16 l21) (road l17 l1) (road l17 l12) (road l17 l16) (road l18 l23) (road l19 l60) (road l2 l25) (road l2 l7) (road l20 l25) (road l20 l26) (road l21 l16) (road l21 l50) (road l22 l11) (road l22 l35) (road l23 l18) (road l23 l66) (road l24 l50) (road l24 l67) (road l25 l2) (road l25 l20) (road l26 l20) (road l26 l46) (road l27 l30) (road l27 l52) (road l28 l39) (road l28 l53) (road l29 l53) (road l29 l6) (road l3 l10) (road l3 l57) (road l30 l27) (road l30 l36) (road l31 l33) (road l31 l47) (road l32 l54) (road l32 l55) (road l33 l31) (road l33 l51) (road l34 l37) (road l34 l68) (road l35 l0) (road l35 l22) (road l36 l30) (road l36 l40) (road l37 l34) (road l37 l71) (road l38 l45) (road l38 l71) (road l39 l28) (road l39 l48) (road l39 l70) (road l4 l46) (road l4 l58) (road l40 l36) (road l40 l69) (road l41 l45) (road l41 l49) (road l41 l67) (road l42 l44) (road l42 l57) (road l43 l52) (road l43 l59) (road l44 l42) (road l44 l56) (road l45 l38) (road l45 l41) (road l46 l26) (road l46 l4) (road l47 l11) (road l47 l31) (road l48 l39) (road l48 l5) (road l49 l41) (road l49 l65) (road l5 l13) (road l5 l48) (road l50 l21) (road l50 l24) (road l51 l33) (road l51 l64) (road l52 l27) (road l52 l43) (road l53 l28) (road l53 l29) (road l54 l32) (road l54 l72) (road l55 l32) (road l55 l62) (road l56 l15) (road l56 l44) (road l57 l3) (road l57 l42) (road l58 l14) (road l58 l4) (road l59 l43) (road l59 l9) (road l6 l14) (road l6 l29) (road l60 l19) (road l60 l7) (road l61 l0) (road l61 l62) (road l62 l55) (road l62 l61) (road l63 l66) (road l63 l72) (road l64 l51) (road l64 l69) (road l65 l49) (road l65 l70) (road l66 l23) (road l66 l63) (road l67 l24) (road l67 l41) (road l68 l34) (road l68 l8) (road l69 l40) (road l69 l64) (road l7 l2) (road l7 l60) (road l70 l39) (road l70 l65) (road l71 l37) (road l71 l38) (road l72 l54) (road l72 l63) (road l8 l68) (road l8 l9) (road l9 l59) (road l9 l8) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l23) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l8) (spare-in l9) (vehicle-at l18))
    (:goal (vehicle-at l19))
)