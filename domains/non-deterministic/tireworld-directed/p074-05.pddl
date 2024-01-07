(define (problem tireworld-074-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l8 l9 - location)
    (:init (road l0 l11) (road l0 l70) (road l1 l30) (road l1 l46) (road l10 l3) (road l10 l53) (road l11 l0) (road l11 l59) (road l12 l33) (road l12 l59) (road l13 l44) (road l14 l16) (road l14 l9) (road l15 l44) (road l15 l56) (road l16 l14) (road l16 l28) (road l17 l25) (road l17 l63) (road l18 l39) (road l18 l63) (road l19 l40) (road l19 l41) (road l2 l40) (road l2 l64) (road l20 l62) (road l20 l68) (road l21 l24) (road l21 l58) (road l22 l27) (road l22 l39) (road l23 l4) (road l23 l61) (road l24 l21) (road l24 l44) (road l25 l17) (road l25 l6) (road l26 l5) (road l26 l60) (road l27 l22) (road l27 l52) (road l28 l16) (road l28 l71) (road l29 l30) (road l29 l36) (road l3 l10) (road l3 l41) (road l3 l55) (road l30 l1) (road l30 l29) (road l31 l32) (road l31 l56) (road l32 l31) (road l32 l35) (road l32 l62) (road l33 l12) (road l33 l69) (road l34 l45) (road l34 l71) (road l35 l32) (road l35 l64) (road l36 l29) (road l36 l70) (road l37 l57) (road l37 l61) (road l38 l55) (road l38 l56) (road l39 l18) (road l39 l22) (road l4 l23) (road l4 l56) (road l40 l19) (road l40 l2) (road l41 l19) (road l41 l3) (road l42 l57) (road l42 l67) (road l43 l65) (road l43 l9) (road l44 l13) (road l44 l15) (road l44 l24) (road l45 l34) (road l45 l49) (road l46 l1) (road l46 l8) (road l47 l48) (road l47 l58) (road l48 l47) (road l48 l51) (road l49 l45) (road l49 l7) (road l5 l26) (road l5 l65) (road l50 l52) (road l50 l72) (road l51 l48) (road l51 l73) (road l52 l27) (road l52 l50) (road l53 l10) (road l53 l6) (road l54 l7) (road l55 l3) (road l55 l38) (road l56 l15) (road l56 l31) (road l56 l38) (road l56 l4) (road l57 l37) (road l57 l42) (road l58 l21) (road l58 l47) (road l59 l11) (road l59 l12) (road l6 l25) (road l6 l53) (road l60 l26) (road l60 l68) (road l60 l69) (road l61 l23) (road l61 l37) (road l62 l20) (road l62 l32) (road l63 l17) (road l63 l18) (road l64 l2) (road l64 l35) (road l65 l43) (road l65 l5) (road l66 l72) (road l66 l8) (road l67 l42) (road l67 l73) (road l68 l20) (road l68 l60) (road l69 l33) (road l69 l60) (road l7 l49) (road l7 l54) (road l70 l0) (road l70 l36) (road l71 l28) (road l71 l34) (road l72 l50) (road l72 l66) (road l73 l51) (road l73 l67) (road l8 l46) (road l8 l66) (road l9 l14) (road l9 l43) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l33) (spare-in l34) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l8) (spare-in l9) (vehicle-at l54))
    (:goal (vehicle-at l13))
)