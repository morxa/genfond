(define (problem tireworld-078-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l8 l9 - location)
    (:init (road l0 l1) (road l1 l26) (road l10 l55) (road l11 l19) (road l11 l5) (road l12 l46) (road l13 l17) (road l14 l17) (road l14 l72) (road l15 l22) (road l15 l23) (road l16 l28) (road l17 l14) (road l18 l51) (road l19 l11) (road l19 l25) (road l2 l7) (road l20 l52) (road l20 l60) (road l21 l31) (road l22 l3) (road l22 l54) (road l23 l15) (road l24 l32) (road l24 l72) (road l25 l65) (road l26 l1) (road l26 l58) (road l26 l8) (road l27 l0) (road l28 l70) (road l29 l33) (road l3 l22) (road l3 l34) (road l30 l18) (road l31 l69) (road l32 l12) (road l32 l24) (road l33 l57) (road l33 l9) (road l35 l47) (road l36 l43) (road l36 l49) (road l37 l64) (road l38 l64) (road l39 l61) (road l39 l73) (road l4 l56) (road l4 l7) (road l40 l27) (road l40 l68) (road l41 l0) (road l42 l49) (road l43 l36) (road l43 l74) (road l44 l48) (road l44 l53) (road l45 l49) (road l46 l5) (road l47 l67) (road l48 l45) (road l49 l36) (road l5 l11) (road l50 l41) (road l51 l61) (road l51 l75) (road l52 l10) (road l52 l20) (road l52 l50) (road l53 l44) (road l54 l68) (road l55 l10) (road l55 l7) (road l56 l4) (road l56 l42) (road l57 l52) (road l58 l53) (road l58 l77) (road l59 l37) (road l6 l2) (road l6 l52) (road l60 l20) (road l60 l39) (road l61 l39) (road l62 l59) (road l63 l16) (road l64 l30) (road l64 l38) (road l65 l63) (road l66 l26) (road l66 l75) (road l67 l13) (road l68 l40) (road l69 l23) (road l7 l2) (road l7 l4) (road l70 l62) (road l71 l6) (road l71 l8) (road l72 l24) (road l73 l74) (road l74 l38) (road l74 l73) (road l75 l66) (road l76 l35) (road l77 l21) (road l8 l71) (road l9 l76) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l73) (spare-in l77) (spare-in l8) (spare-in l9) (vehicle-at l29))
    (:goal (vehicle-at l34))
)