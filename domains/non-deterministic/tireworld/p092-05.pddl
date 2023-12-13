(define (problem tireworld-092-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 - location)
    (:init (road l0 l49) (road l0 l52) (road l1 l65) (road l1 l86) (road l10 l5) (road l10 l65) (road l11 l37) (road l11 l9) (road l12 l43) (road l12 l84) (road l13 l31) (road l13 l48) (road l14 l17) (road l14 l28) (road l15 l36) (road l15 l68) (road l16 l36) (road l16 l77) (road l17 l14) (road l17 l30) (road l18 l19) (road l18 l22) (road l18 l4) (road l18 l57) (road l18 l69) (road l19 l18) (road l19 l5) (road l2 l29) (road l2 l80) (road l20 l27) (road l20 l73) (road l21 l44) (road l21 l90) (road l22 l18) (road l22 l76) (road l23 l28) (road l23 l89) (road l24 l37) (road l24 l52) (road l25 l72) (road l25 l81) (road l26 l57) (road l26 l75) (road l26 l8) (road l27 l20) (road l27 l60) (road l28 l14) (road l28 l23) (road l29 l2) (road l29 l87) (road l3 l33) (road l3 l54) (road l30 l17) (road l30 l84) (road l31 l13) (road l31 l85) (road l32 l44) (road l32 l71) (road l33 l3) (road l33 l42) (road l34 l51) (road l34 l85) (road l35 l64) (road l35 l89) (road l36 l15) (road l36 l16) (road l37 l11) (road l37 l24) (road l37 l4) (road l38 l46) (road l38 l83) (road l39 l40) (road l39 l70) (road l4 l18) (road l4 l37) (road l40 l39) (road l40 l74) (road l41 l79) (road l41 l83) (road l42 l33) (road l42 l82) (road l43 l12) (road l43 l71) (road l44 l21) (road l44 l32) (road l45 l66) (road l45 l91) (road l46 l38) (road l46 l86) (road l47 l50) (road l47 l78) (road l48 l13) (road l48 l7) (road l49 l0) (road l49 l67) (road l5 l10) (road l5 l19) (road l5 l6) (road l50 l47) (road l50 l90) (road l51 l34) (road l51 l77) (road l52 l0) (road l52 l24) (road l53 l55) (road l53 l62) (road l54 l3) (road l54 l55) (road l55 l53) (road l55 l54) (road l56 l69) (road l56 l76) (road l57 l18) (road l57 l26) (road l58 l61) (road l59 l60) (road l59 l63) (road l6 l5) (road l6 l87) (road l60 l27) (road l60 l59) (road l61 l58) (road l61 l76) (road l62 l53) (road l62 l70) (road l63 l59) (road l63 l81) (road l64 l35) (road l64 l76) (road l65 l1) (road l65 l10) (road l66 l45) (road l66 l72) (road l67 l49) (road l68 l15) (road l68 l9) (road l69 l18) (road l69 l56) (road l7 l48) (road l7 l73) (road l70 l39) (road l70 l62) (road l71 l32) (road l71 l43) (road l72 l25) (road l72 l66) (road l73 l20) (road l73 l7) (road l74 l40) (road l74 l80) (road l75 l26) (road l75 l89) (road l76 l22) (road l76 l56) (road l76 l61) (road l76 l64) (road l76 l8) (road l77 l16) (road l77 l51) (road l78 l47) (road l78 l82) (road l79 l41) (road l79 l88) (road l8 l26) (road l8 l76) (road l80 l2) (road l80 l74) (road l81 l25) (road l81 l63) (road l82 l42) (road l82 l78) (road l83 l38) (road l83 l41) (road l84 l12) (road l84 l30) (road l85 l31) (road l85 l34) (road l86 l1) (road l86 l46) (road l87 l29) (road l87 l6) (road l88 l79) (road l88 l91) (road l89 l23) (road l89 l35) (road l89 l75) (road l9 l11) (road l9 l68) (road l90 l21) (road l90 l50) (road l91 l45) (road l91 l88) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (vehicle-at l67))
    (:goal (vehicle-at l58))
)