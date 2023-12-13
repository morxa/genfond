(define (problem tireworld-076-09)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l3) (road l0 l8) (road l1 l40) (road l1 l67) (road l10 l30) (road l10 l71) (road l11 l48) (road l11 l71) (road l12 l53) (road l12 l67) (road l13 l46) (road l14 l22) (road l14 l39) (road l15 l43) (road l15 l62) (road l16 l26) (road l16 l52) (road l17 l36) (road l17 l68) (road l18 l27) (road l18 l47) (road l19 l63) (road l19 l68) (road l2 l52) (road l2 l70) (road l20 l37) (road l20 l50) (road l21 l36) (road l21 l37) (road l22 l14) (road l22 l69) (road l22 l7) (road l22 l73) (road l23 l40) (road l23 l56) (road l24 l30) (road l24 l46) (road l25 l4) (road l25 l73) (road l26 l16) (road l26 l44) (road l27 l18) (road l27 l5) (road l28 l49) (road l28 l59) (road l29 l50) (road l29 l57) (road l3 l0) (road l3 l58) (road l30 l10) (road l30 l24) (road l31 l43) (road l31 l54) (road l32 l6) (road l32 l61) (road l33 l42) (road l33 l64) (road l34 l48) (road l34 l49) (road l35 l5) (road l35 l58) (road l36 l17) (road l36 l21) (road l37 l20) (road l37 l21) (road l38 l43) (road l38 l69) (road l39 l14) (road l39 l61) (road l4 l25) (road l4 l63) (road l40 l1) (road l40 l23) (road l41 l66) (road l42 l33) (road l42 l72) (road l43 l15) (road l43 l31) (road l43 l38) (road l43 l57) (road l44 l26) (road l44 l51) (road l45 l64) (road l45 l9) (road l46 l13) (road l46 l24) (road l47 l18) (road l47 l75) (road l48 l11) (road l48 l34) (road l49 l28) (road l49 l34) (road l5 l27) (road l5 l35) (road l50 l20) (road l50 l29) (road l51 l44) (road l51 l60) (road l52 l16) (road l52 l2) (road l53 l12) (road l53 l65) (road l54 l31) (road l54 l61) (road l55 l72) (road l55 l75) (road l56 l23) (road l56 l66) (road l57 l29) (road l57 l43) (road l58 l3) (road l58 l35) (road l59 l28) (road l59 l7) (road l6 l32) (road l6 l72) (road l60 l51) (road l60 l65) (road l61 l32) (road l61 l39) (road l61 l54) (road l62 l15) (road l62 l74) (road l63 l19) (road l63 l4) (road l64 l33) (road l64 l45) (road l65 l53) (road l65 l60) (road l66 l41) (road l66 l56) (road l67 l1) (road l67 l12) (road l68 l17) (road l68 l19) (road l69 l22) (road l69 l38) (road l7 l22) (road l7 l59) (road l70 l2) (road l70 l8) (road l71 l10) (road l71 l11) (road l72 l42) (road l72 l55) (road l72 l6) (road l73 l22) (road l73 l25) (road l74 l62) (road l74 l9) (road l75 l47) (road l75 l55) (road l8 l0) (road l8 l70) (road l9 l45) (road l9 l74) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l8) (spare-in l9) (vehicle-at l41))
    (:goal (vehicle-at l13))
)