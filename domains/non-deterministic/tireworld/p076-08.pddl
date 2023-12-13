(define (problem tireworld-076-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l57) (road l0 l67) (road l1 l19) (road l1 l71) (road l10 l5) (road l10 l67) (road l11 l12) (road l11 l57) (road l12 l11) (road l12 l51) (road l13 l43) (road l13 l68) (road l14 l30) (road l14 l41) (road l15 l39) (road l15 l67) (road l16 l47) (road l16 l59) (road l17 l45) (road l17 l5) (road l17 l7) (road l18 l34) (road l18 l44) (road l18 l62) (road l19 l1) (road l19 l50) (road l2 l41) (road l2 l56) (road l20 l33) (road l20 l45) (road l21 l32) (road l21 l74) (road l22 l23) (road l23 l22) (road l23 l39) (road l24 l3) (road l24 l4) (road l25 l6) (road l25 l69) (road l26 l68) (road l26 l75) (road l27 l36) (road l27 l52) (road l28 l4) (road l28 l61) (road l29 l38) (road l29 l44) (road l3 l24) (road l3 l73) (road l3 l74) (road l30 l14) (road l30 l65) (road l31 l37) (road l32 l21) (road l32 l50) (road l33 l20) (road l33 l65) (road l34 l18) (road l34 l71) (road l35 l49) (road l35 l58) (road l36 l27) (road l36 l48) (road l37 l31) (road l37 l75) (road l38 l29) (road l38 l70) (road l39 l15) (road l39 l23) (road l4 l24) (road l4 l28) (road l40 l66) (road l40 l70) (road l41 l14) (road l41 l2) (road l42 l53) (road l42 l72) (road l43 l13) (road l43 l54) (road l44 l18) (road l44 l29) (road l45 l17) (road l45 l20) (road l46 l7) (road l46 l71) (road l47 l16) (road l47 l51) (road l48 l36) (road l48 l8) (road l49 l35) (road l49 l62) (road l5 l10) (road l5 l17) (road l50 l19) (road l50 l32) (road l51 l12) (road l51 l47) (road l52 l27) (road l52 l72) (road l53 l42) (road l53 l54) (road l54 l43) (road l54 l53) (road l55 l6) (road l55 l60) (road l56 l2) (road l56 l63) (road l57 l0) (road l57 l11) (road l58 l35) (road l58 l59) (road l59 l16) (road l59 l58) (road l6 l25) (road l6 l55) (road l60 l55) (road l60 l9) (road l61 l28) (road l61 l64) (road l62 l18) (road l62 l49) (road l63 l56) (road l63 l69) (road l64 l61) (road l64 l8) (road l65 l30) (road l65 l33) (road l66 l40) (road l66 l73) (road l67 l0) (road l67 l10) (road l67 l15) (road l68 l13) (road l68 l26) (road l69 l25) (road l69 l63) (road l7 l17) (road l7 l46) (road l70 l38) (road l70 l40) (road l71 l1) (road l71 l34) (road l71 l46) (road l71 l9) (road l72 l42) (road l72 l52) (road l73 l3) (road l73 l66) (road l74 l21) (road l74 l3) (road l75 l26) (road l75 l37) (road l8 l48) (road l8 l64) (road l9 l60) (road l9 l71) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l61) (spare-in l62) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l8) (vehicle-at l22))
    (:goal (vehicle-at l31))
)