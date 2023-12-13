(define (problem tireworld-076-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l20) (road l0 l30) (road l1 l43) (road l1 l72) (road l10 l53) (road l10 l62) (road l11 l30) (road l11 l64) (road l12 l44) (road l12 l56) (road l13 l71) (road l13 l74) (road l14 l36) (road l14 l58) (road l15 l23) (road l15 l56) (road l16 l20) (road l16 l42) (road l17 l45) (road l18 l40) (road l18 l50) (road l19 l57) (road l19 l67) (road l2 l22) (road l2 l7) (road l20 l0) (road l20 l16) (road l21 l33) (road l21 l46) (road l22 l2) (road l22 l72) (road l23 l15) (road l23 l51) (road l24 l52) (road l24 l64) (road l25 l32) (road l25 l75) (road l26 l35) (road l26 l67) (road l27 l28) (road l27 l34) (road l28 l27) (road l28 l43) (road l28 l66) (road l29 l32) (road l29 l8) (road l3 l6) (road l3 l63) (road l30 l0) (road l30 l11) (road l31 l37) (road l31 l66) (road l32 l25) (road l32 l29) (road l33 l21) (road l33 l49) (road l34 l27) (road l34 l55) (road l35 l26) (road l35 l73) (road l36 l14) (road l36 l65) (road l37 l31) (road l37 l71) (road l38 l48) (road l38 l60) (road l39 l4) (road l39 l71) (road l4 l39) (road l4 l40) (road l40 l18) (road l40 l4) (road l41 l68) (road l41 l69) (road l42 l16) (road l42 l48) (road l43 l1) (road l43 l28) (road l44 l12) (road l44 l49) (road l45 l17) (road l45 l70) (road l46 l21) (road l46 l50) (road l47 l7) (road l47 l74) (road l48 l38) (road l48 l42) (road l49 l33) (road l49 l44) (road l5 l55) (road l5 l9) (road l50 l18) (road l50 l46) (road l50 l68) (road l51 l23) (road l51 l62) (road l52 l24) (road l52 l53) (road l53 l10) (road l53 l52) (road l54 l55) (road l54 l61) (road l55 l34) (road l55 l5) (road l55 l54) (road l56 l12) (road l56 l15) (road l57 l19) (road l58 l14) (road l58 l59) (road l59 l58) (road l59 l8) (road l6 l3) (road l6 l70) (road l60 l38) (road l60 l65) (road l61 l54) (road l61 l63) (road l62 l10) (road l62 l51) (road l63 l3) (road l63 l61) (road l64 l11) (road l64 l24) (road l65 l36) (road l65 l60) (road l66 l28) (road l66 l31) (road l67 l19) (road l67 l26) (road l68 l41) (road l68 l50) (road l69 l41) (road l69 l73) (road l7 l2) (road l7 l47) (road l70 l45) (road l70 l6) (road l71 l13) (road l71 l37) (road l71 l39) (road l72 l1) (road l72 l22) (road l73 l35) (road l73 l69) (road l74 l13) (road l74 l47) (road l75 l25) (road l75 l9) (road l8 l29) (road l8 l59) (road l9 l5) (road l9 l75) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l8) (spare-in l9) (vehicle-at l57))
    (:goal (vehicle-at l17))
)