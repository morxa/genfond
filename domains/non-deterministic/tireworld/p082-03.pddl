(define (problem tireworld-082-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l9 - location)
    (:init (road l0 l15) (road l0 l62) (road l1 l28) (road l1 l80) (road l10 l37) (road l10 l76) (road l11 l28) (road l11 l44) (road l12 l45) (road l12 l58) (road l12 l68) (road l13 l14) (road l13 l29) (road l13 l42) (road l14 l13) (road l14 l51) (road l15 l0) (road l15 l32) (road l16 l30) (road l16 l77) (road l17 l40) (road l17 l63) (road l17 l73) (road l18 l55) (road l18 l9) (road l19 l41) (road l19 l46) (road l2 l65) (road l2 l77) (road l20 l23) (road l20 l33) (road l21 l38) (road l21 l7) (road l22 l4) (road l22 l47) (road l23 l20) (road l23 l64) (road l24 l51) (road l24 l6) (road l25 l57) (road l25 l66) (road l26 l53) (road l26 l55) (road l27 l40) (road l27 l41) (road l28 l1) (road l28 l11) (road l29 l13) (road l3 l48) (road l3 l68) (road l30 l16) (road l30 l56) (road l30 l7) (road l31 l44) (road l31 l50) (road l32 l15) (road l32 l39) (road l33 l20) (road l33 l50) (road l34 l67) (road l34 l81) (road l35 l58) (road l35 l63) (road l35 l75) (road l36 l71) (road l36 l79) (road l37 l10) (road l37 l80) (road l38 l21) (road l38 l39) (road l38 l70) (road l39 l32) (road l39 l38) (road l4 l22) (road l4 l6) (road l4 l73) (road l4 l9) (road l40 l17) (road l40 l27) (road l40 l45) (road l41 l19) (road l41 l27) (road l42 l13) (road l42 l60) (road l43 l49) (road l43 l71) (road l44 l11) (road l44 l31) (road l45 l12) (road l45 l40) (road l46 l19) (road l46 l76) (road l47 l22) (road l47 l70) (road l48 l3) (road l48 l54) (road l48 l77) (road l49 l43) (road l49 l61) (road l5 l72) (road l5 l75) (road l50 l31) (road l50 l33) (road l51 l14) (road l51 l24) (road l52 l78) (road l52 l81) (road l53 l26) (road l53 l56) (road l54 l48) (road l54 l60) (road l55 l18) (road l55 l26) (road l56 l30) (road l56 l53) (road l57 l25) (road l57 l78) (road l58 l12) (road l58 l35) (road l59 l65) (road l59 l72) (road l6 l24) (road l6 l4) (road l60 l42) (road l60 l54) (road l61 l49) (road l61 l67) (road l62 l0) (road l62 l8) (road l63 l17) (road l63 l35) (road l64 l23) (road l64 l69) (road l65 l2) (road l65 l59) (road l66 l25) (road l66 l69) (road l67 l34) (road l67 l61) (road l68 l12) (road l68 l3) (road l69 l64) (road l69 l66) (road l7 l21) (road l7 l30) (road l70 l38) (road l70 l47) (road l71 l36) (road l71 l43) (road l72 l5) (road l72 l59) (road l73 l17) (road l73 l4) (road l74 l75) (road l74 l79) (road l74 l8) (road l75 l35) (road l75 l5) (road l75 l74) (road l76 l10) (road l76 l46) (road l77 l16) (road l77 l2) (road l77 l48) (road l78 l52) (road l78 l57) (road l79 l36) (road l79 l74) (road l8 l62) (road l8 l74) (road l80 l1) (road l80 l37) (road l81 l34) (road l81 l52) (road l9 l18) (road l9 l4) (spare-in l0) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l25) (spare-in l28) (spare-in l3) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l58) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l73) (spare-in l74) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l9) (vehicle-at l38))
    (:goal (vehicle-at l29))
)