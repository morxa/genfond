(define (problem tireworld-078-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l8 l9 - location)
    (:init (road l0 l31) (road l0 l38) (road l1 l21) (road l1 l71) (road l10 l54) (road l10 l74) (road l11 l42) (road l11 l52) (road l12 l14) (road l12 l77) (road l13 l34) (road l13 l41) (road l14 l12) (road l14 l56) (road l15 l55) (road l15 l70) (road l16 l27) (road l16 l31) (road l16 l57) (road l17 l27) (road l17 l61) (road l18 l39) (road l18 l59) (road l19 l9) (road l2 l27) (road l2 l44) (road l2 l57) (road l2 l65) (road l20 l47) (road l20 l66) (road l21 l1) (road l21 l54) (road l22 l30) (road l22 l65) (road l23 l35) (road l23 l40) (road l24 l51) (road l24 l8) (road l25 l5) (road l25 l67) (road l26 l60) (road l26 l76) (road l27 l16) (road l27 l17) (road l27 l2) (road l27 l6) (road l28 l33) (road l28 l48) (road l29 l3) (road l29 l46) (road l3 l29) (road l3 l43) (road l30 l22) (road l30 l35) (road l31 l0) (road l31 l16) (road l32 l51) (road l32 l72) (road l33 l28) (road l33 l56) (road l34 l13) (road l34 l53) (road l35 l23) (road l35 l30) (road l36 l60) (road l36 l61) (road l37 l42) (road l37 l75) (road l38 l0) (road l39 l18) (road l39 l41) (road l4 l45) (road l4 l7) (road l40 l23) (road l40 l50) (road l41 l13) (road l41 l39) (road l42 l11) (road l42 l37) (road l43 l3) (road l43 l68) (road l44 l2) (road l44 l9) (road l45 l4) (road l45 l69) (road l46 l29) (road l46 l47) (road l46 l62) (road l47 l20) (road l47 l46) (road l48 l28) (road l48 l75) (road l49 l71) (road l49 l72) (road l5 l25) (road l5 l73) (road l50 l40) (road l50 l64) (road l51 l24) (road l51 l32) (road l52 l11) (road l52 l69) (road l53 l34) (road l53 l63) (road l54 l10) (road l54 l21) (road l55 l15) (road l55 l58) (road l56 l14) (road l56 l33) (road l57 l16) (road l57 l2) (road l58 l55) (road l58 l62) (road l59 l18) (road l59 l7) (road l6 l27) (road l6 l73) (road l60 l26) (road l60 l36) (road l61 l17) (road l61 l36) (road l62 l46) (road l62 l58) (road l63 l53) (road l63 l66) (road l64 l50) (road l64 l77) (road l65 l2) (road l65 l22) (road l66 l20) (road l66 l63) (road l67 l25) (road l67 l8) (road l68 l43) (road l68 l74) (road l69 l45) (road l69 l52) (road l7 l4) (road l7 l59) (road l70 l15) (road l70 l76) (road l71 l1) (road l71 l49) (road l72 l32) (road l72 l49) (road l73 l5) (road l73 l6) (road l74 l10) (road l74 l68) (road l75 l37) (road l75 l48) (road l76 l26) (road l76 l70) (road l77 l12) (road l77 l64) (road l8 l24) (road l8 l67) (road l9 l19) (road l9 l44) (spare-in l0) (spare-in l1) (spare-in l16) (spare-in l2) (spare-in l27) (spare-in l31) (spare-in l39) (spare-in l42) (spare-in l44) (spare-in l49) (spare-in l54) (spare-in l68) (spare-in l71) (spare-in l76) (spare-in l9) (vehicle-at l38))
    (:goal (vehicle-at l19))
)