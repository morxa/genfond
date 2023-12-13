(define (problem tireworld-076-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l24) (road l0 l40) (road l1 l58) (road l1 l59) (road l10 l19) (road l10 l25) (road l10 l38) (road l10 l55) (road l11 l3) (road l11 l40) (road l12 l2) (road l12 l34) (road l13 l59) (road l13 l6) (road l14 l21) (road l14 l35) (road l14 l45) (road l15 l21) (road l15 l42) (road l16 l22) (road l16 l56) (road l17 l47) (road l17 l64) (road l18 l29) (road l18 l4) (road l19 l10) (road l19 l22) (road l19 l50) (road l2 l12) (road l2 l62) (road l20 l25) (road l20 l61) (road l21 l14) (road l21 l15) (road l22 l16) (road l22 l19) (road l22 l68) (road l22 l71) (road l22 l72) (road l23 l28) (road l23 l7) (road l24 l0) (road l24 l49) (road l25 l10) (road l25 l20) (road l26 l42) (road l27 l61) (road l28 l23) (road l28 l54) (road l29 l18) (road l29 l9) (road l3 l11) (road l3 l69) (road l30 l36) (road l30 l44) (road l31 l41) (road l31 l47) (road l32 l37) (road l32 l54) (road l33 l58) (road l33 l70) (road l34 l12) (road l34 l38) (road l35 l14) (road l35 l66) (road l36 l30) (road l36 l53) (road l37 l32) (road l37 l60) (road l38 l10) (road l38 l34) (road l39 l57) (road l39 l63) (road l4 l18) (road l4 l46) (road l40 l0) (road l40 l11) (road l41 l31) (road l41 l62) (road l42 l15) (road l42 l26) (road l43 l49) (road l43 l71) (road l44 l30) (road l44 l74) (road l45 l14) (road l45 l46) (road l46 l4) (road l46 l45) (road l47 l17) (road l47 l31) (road l48 l74) (road l48 l8) (road l49 l24) (road l49 l43) (road l5 l56) (road l5 l66) (road l50 l19) (road l50 l53) (road l51 l57) (road l51 l69) (road l52 l68) (road l52 l8) (road l53 l36) (road l53 l50) (road l54 l28) (road l54 l32) (road l55 l10) (road l55 l74) (road l56 l16) (road l56 l5) (road l57 l39) (road l57 l51) (road l58 l1) (road l58 l33) (road l59 l1) (road l59 l13) (road l6 l13) (road l6 l75) (road l60 l37) (road l60 l70) (road l61 l20) (road l61 l27) (road l62 l2) (road l62 l41) (road l63 l39) (road l63 l67) (road l64 l17) (road l64 l65) (road l65 l64) (road l65 l7) (road l66 l35) (road l66 l5) (road l67 l63) (road l67 l75) (road l68 l22) (road l68 l52) (road l69 l3) (road l69 l51) (road l7 l23) (road l7 l65) (road l70 l33) (road l70 l60) (road l71 l22) (road l71 l43) (road l72 l22) (road l72 l73) (road l73 l72) (road l73 l9) (road l74 l44) (road l74 l48) (road l74 l55) (road l75 l6) (road l75 l67) (road l8 l48) (road l8 l52) (road l9 l29) (road l9 l73) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l51) (spare-in l54) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l8) (spare-in l9) (vehicle-at l26))
    (:goal (vehicle-at l27))
)